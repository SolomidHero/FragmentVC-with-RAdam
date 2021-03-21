#!/usr/bin/env python3
"""Train FragmentVC model."""

# import argparse
import datetime
import random
from pathlib import Path
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_optimizer import RAdam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from jsonargparse import ArgumentParser, ActionConfigFile

from data import IntraSpeakerDataset, collate_batch, infinite_iterator, get_mel_plot
from models import (
    FragmentVC, Discriminator, ConditionalDiscriminator, load_pretrained_spk_emb,
    adversarial_loss, discriminator_loss, mel_spec_loss, cosine_sim_loss,
    get_cosine_schedule_with_warmup
)


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--total_steps", type=int, default=60000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--valid_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--milestones", type=int, nargs=2, default=[15000, 30000])
    parser.add_argument("--exclusive_rate", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--accu_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--grad_norm_clip", type=float, default=10.0)
    parser.add_argument("--use_target_features", action='store_true')
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--d_ckpt", type=str, default=None)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--sim_ckpt", type=str, default=None)
    parser.add_argument("--d_sim_ckpt", type=str, default=None)
    parser.add_argument("--train_config", action=ActionConfigFile)
    return vars(parser.parse_args())

use_embed_in_src = True

def disc_fn(disc, batch, mask):
    scores = torch.cat([
        torch.flatten(disc(torch.masked_select(item, item_mask).reshape(1, item.shape[0], -1)))
        for item, item_mask in zip(batch, mask)
    ])
    return scores

def sim_fn(sim_model, batch, mask):
    scores = torch.cat([
        sim_model(torch.masked_select(item, item_mask).reshape(1, item.shape[0], -1))
        for item, item_mask in zip(batch, mask)
    ])
    return scores

def sim_disc_fn(sim_disc, batch, conds, mask):
    scores = torch.cat([
        torch.flatten(sim_disc(
            torch.masked_select(item, item_mask).reshape(1, item.shape[0], -1),
            cond.unsqueeze(0),
        ))
        for item, cond, item_mask in zip(batch, conds, mask)
    ])
    return scores

def model_fn(batch, model, self_exclude, ref_included, device, cross=False):
    """Forward a batch through model."""

    srcs, src_masks, (refs, refs_features), ref_masks, tgts, tgt_masks, tgt_spk_embs, _ = batch

    srcs = srcs.to(device)
    src_masks = src_masks.to(device)
    refs = refs.to(device)
    refs_features = refs_features.to(device) if refs_features is not None else refs_features
    ref_masks = ref_masks.to(device)
    tgts = tgts.to(device)
    tgt_masks = tgt_masks.to(device)
    tgt_spk_embs = None if tgt_spk_embs is None else tgt_spk_embs.to(device)

    if ref_included:
        if random.random() >= self_exclude:
            refs = torch.cat((refs, tgts), dim=2)
            refs_features = torch.cat((refs_features, srcs), dim=1) if refs_features is not None else refs_features
            ref_masks = torch.cat((ref_masks, tgt_masks), dim=1)
    else:
        refs = tgts
        refs_features = srcs if refs_features is not None else refs_features
        ref_masks = tgt_masks

    if cross:
        srcs = torch.roll(srcs, 1, 0)
        src_masks = torch.roll(src_masks, 1, 0)
    if use_embed_in_src:
        # (batch_len, time, features_dim)
        cond = torch.repeat_interleave(tgt_spk_embs, srcs.shape[1], dim=0).reshape(*srcs.shape)
        srcs = torch.cat([srcs, cond], dim=2)
    outs, _ = model(srcs, refs, refs_features=refs_features, src_masks=src_masks, ref_masks=ref_masks)
    return outs, tgts, tgt_spk_embs, ~src_masks


def training_step(batches, model, g_optimizer=None, g_scheduler=None, disc=None, d_optimizer=None, d_scheduler=None,
        sim_model=None, sim_disc=None, device='cuda',
        grad_norm_clip=10, self_exclude=1.0, ref_included=True
    ):

    # loss coefficients
    adv_lambda = 0.03
    cond_lambda = 0.02
    sim_lambda = 0.02

    # Discriminator losses
    d_loss = torch.tensor(0., device=device)
    if disc is not None:
        for batch in batches:
            # identity
            with torch.no_grad():
                outs, tgts, tgt_spk_embs, mask = model_fn(batch, model, self_exclude, ref_included, device)
            fake_scores, real_scores = disc_fn(disc, outs, mask), disc_fn(disc, tgts, mask)
            d_loss += adv_lambda * discriminator_loss(fake_scores, real_scores)
            if sim_disc is not None:
                d_loss += cond_lambda * discriminator_loss(sim_disc_fn(sim_disc, outs, tgt_spk_embs, mask), sim_disc_fn(sim_disc, tgts, tgt_spk_embs, mask))

            # cross-conversion
            with torch.no_grad():
                outs, tgts, tgt_spk_embs, mask = model_fn(batch, model, self_exclude, ref_included, device, cross=True)
            fake_scores, real_scores = disc_fn(disc, outs, mask), disc_fn(disc, tgts, mask)
            d_loss += adv_lambda * discriminator_loss(fake_scores, real_scores)
            if sim_disc is not None:
                d_loss += cond_lambda * discriminator_loss(sim_disc_fn(sim_disc, outs, tgt_spk_embs, mask), sim_disc_fn(sim_disc, tgts, tgt_spk_embs, mask))

        d_loss /= len(batches)

        d_optimizer.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_norm_clip)
        if sim_disc is not None:
            torch.nn.utils.clip_grad_norm_(sim_disc.parameters(), grad_norm_clip)
        d_optimizer.step()

    # Generator losses
    g_loss = torch.tensor(0., device=device)
    g_adv_loss = torch.tensor(0., device=device)
    sim_loss = torch.tensor(0., device=device)
    for batch in batches:
        # identity
        outs, tgts, tgt_spk_embs, mask = model_fn(batch, model, self_exclude, ref_included, device)
        g_loss += mel_spec_loss(outs, tgts, mask=mask)
        if disc is not None:
            g_adv_loss += adv_lambda * adversarial_loss(disc_fn(disc, outs, mask))
        if sim_model is not None:
            sim_loss += sim_lambda * cosine_sim_loss(sim_fn(sim_model, outs, mask), tgt_spk_embs)
        if sim_disc is not None:
            g_adv_loss += cond_lambda * adversarial_loss(sim_disc_fn(sim_disc, outs, tgt_spk_embs, mask))

        # cross-conversion
        if sim_model is not None or disc is not None:
            outs, tgts, tgt_spk_embs, mask = model_fn(batch, model, self_exclude, ref_included, device, cross=True)
        if disc is not None:
            g_adv_loss += adv_lambda * adversarial_loss(disc_fn(disc, outs, mask))
        if sim_model is not None:
            sim_loss += sim_lambda * cosine_sim_loss(sim_fn(sim_model, outs, mask), tgt_spk_embs)
        if sim_disc is not None:
            g_adv_loss += cond_lambda * adversarial_loss(sim_disc_fn(sim_disc, outs, tgt_spk_embs, mask))

    sim_loss /= len(batches)
    g_adv_loss /= len(batches)
    g_loss = g_loss / len(batches) + g_adv_loss + sim_loss

    if g_optimizer is not None:
        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        if sim_model is not None:
            torch.nn.utils.clip_grad_norm_(sim_model.parameters(), grad_norm_clip)
        g_optimizer.step()


    if d_scheduler is not None:
        d_scheduler.step()
    if g_scheduler is not None:
        g_scheduler.step()

    return g_loss.item(), g_adv_loss.item(), sim_loss.item(), d_loss.item()


def valid(dataloader, model, device, writer=None):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            outs, tgts, _ = model_fn(batch, model, 1.0, True, device)
            running_loss += mel_spec_loss(outs, tgts).item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(loss=f"{running_loss / (i+1):.2f}")


    if writer is not None:
        for i, _ in zip(range(4), range(len(outs))):
            writer.add_figure('ground_truth/spec_{}'.format(i), get_mel_plot(outs[i]), 1)
            writer.add_figure('converted/spec_{}'.format(i), get_mel_plot(tgts[i]), 1)

    pbar.close()
    model.train()

    return running_loss / len(dataloader)


def main(
    data_dir,
    save_dir,
    total_steps,
    warmup_steps,
    valid_steps,
    log_steps,
    save_steps,
    milestones,
    exclusive_rate,
    n_samples,
    accu_steps,
    batch_size,
    n_workers,
    preload,
    comment,
    ckpt,
    grad_norm_clip,
    use_target_features,
    adv,
    d_ckpt,
    sim,
    sim_ckpt,
    d_sim_ckpt,
    **kwargs,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_path = Path(data_dir) / "metadata.json"

    dataset = IntraSpeakerDataset(data_dir, metadata_path, n_samples, preload, ref_feat=use_target_features)
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)
    train_loader = infinite_iterator(DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    ))
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size * accu_steps,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    if comment is not None:
        log_dir = "logs/"
        log_dir += datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_dir += "_" + comment
        writer = SummaryWriter(log_dir)

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    has_disc = adv or d_ckpt is not None
    disc = None
    sim_disc = None
    sim_model = None
    d_optimizer = None
    d_scheduler = None
    if has_disc:
        if d_ckpt is None:
            disc = Discriminator().to(device)
            disc = torch.jit.script(disc)
        else:
            disc = torch.jit.load(d_ckpt).to(device)

        if d_sim_ckpt is None:
            sim_disc = ConditionalDiscriminator().to(device)
            sim_disc = torch.jit.script(sim_disc)
        else:
            sim_disc = torch.jit.load(d_sim_ckpt).to(device)

        d_optimizer = RAdam(chain(disc.parameters(), [] if sim_disc is None else sim_disc.parameters()), lr=1e-4)
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.99997)

    has_sim = sim or sim_ckpt is not None
    if has_sim:
        sim_model = load_pretrained_spk_emb(train=True).to(device)
        if sim_ckpt is not None:
            sim_model.load_state_dict(torch.load(sim_ckpt))

    if ckpt is not None:
        try:
            start_step = int(ckpt.split('-')[1][4:])
            ref_included = True
        except:
            start_step = 0
            ref_included = False

        model = torch.jit.load(ckpt).to(device)
        g_optimizer = RAdam([
            {"params": model.unet.parameters(), "lr": 1e-4 if start_step < milestones[0] else 1e-6},
            {"params": model.smoothers.parameters()},
            {"params": model.mel_linear.parameters()},
            {"params": model.post_net.parameters()},
            {"params": sim_model.parameters() if has_sim else []},
        ], lr=1e-4,
        )
        g_scheduler = get_cosine_schedule_with_warmup(
            g_optimizer, warmup_steps, total_steps - start_step
        )
        print("Optimizer and scheduler restarted.")
        print(f"Model loaded from {ckpt}, iteration: {start_step}")
    else:
        ref_included = False
        start_step = 0

        model = FragmentVC().to(device)
        model = torch.jit.script(model)
        g_optimizer = RAdam(chain(model.parameters(), sim_model.parameters()), lr=1e-4)
        g_scheduler = get_cosine_schedule_with_warmup(g_optimizer, warmup_steps, total_steps)


    self_exclude = 0.0

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(start_step, total_steps):
        batch_loss = 0.0

        batches = [next(train_loader) for _ in range(accu_steps)]

        batch_loss, g_adv_loss, sim_loss, d_loss = training_step(
            batches, model, g_optimizer, g_scheduler, disc, d_optimizer, d_scheduler, sim_model=sim_model, sim_disc=sim_disc,
            device=device, grad_norm_clip=grad_norm_clip, self_exclude=self_exclude, ref_included=ref_included
        )

        pbar.update()
        pbar.set_postfix(loss=f"{batch_loss:.2f}", excl=self_exclude, step=step + 1)

        if step % log_steps == 0 and comment is not None:
            writer.add_scalar("Loss/train", batch_loss, step)
            writer.add_scalar("Loss/g_adv", g_adv_loss, step)
            writer.add_scalar("Loss/g_similarity", sim_loss, step)
            writer.add_scalar("Loss/d_loss", d_loss, step)
            writer.add_scalar("Self-exclusive Rate", self_exclude, step)

        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_loss = valid(valid_loader, model, device, writer=None if comment is None else writer)

            if comment is not None:
                writer.add_scalar("Loss/valid", valid_loss, step + 1)

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0:
            loss_str = f"{valid_loss:.4f}".replace(".", "dot")
            curr_ckpt_name = f"retriever-step{step+1}-loss{loss_str}.pt"
            curr_d_ckpt_name = f"d_retriever-step{step+1}-loss{loss_str}.pt"
            curr_d_sim_ckpt_name = f"dsim_retriever-step{step+1}-loss{loss_str}.pt"
            curr_sim_ckpt_name = f"sim_retriever-step{step+1}-loss{loss_str}.pt"

            model.cpu()
            model.save(str(save_dir_path / curr_ckpt_name))
            model.to(device)

            if disc is not None:
                disc.cpu()
                disc.save(str(save_dir_path / curr_d_ckpt_name))
                disc.to(device)

            if sim_disc is not None:
                sim_disc.cpu()
                sim_disc.save(str(save_dir_path / curr_d_sim_ckpt_name))
                sim_disc.to(device)

            if has_sim:
                sim_model.cpu()
                torch.save(sim_model.state_dict(), str(save_dir_path / curr_sim_ckpt_name))
                sim_model.to(device)

            pbar.write(f"Step {step + 1} model saved. (loss={valid_loss:.4f})")

        if (step + 1) >= milestones[1]:
            self_exclude = exclusive_rate

        elif (step + 1) == milestones[0]:
            ref_included = True
            g_optimizer = RAdam(
                [
                    {"params": model.unet.parameters(), "lr": 1e-6},
                    {"params": model.smoothers.parameters()},
                    {"params": model.mel_linear.parameters()},
                    {"params": model.post_net.parameters()},
                    {"params": sim_model.parameters() if has_sim else []},
                ],
                lr=1e-4,
            )
            g_scheduler = get_cosine_schedule_with_warmup(
                g_optimizer, warmup_steps, total_steps - milestones[0]
            )
            pbar.write("Optimizer and scheduler restarted.")

        elif (step + 1) > milestones[0]:
            self_exclude = (step + 1 - milestones[0]) / (milestones[1] - milestones[0])
            self_exclude *= exclusive_rate

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
