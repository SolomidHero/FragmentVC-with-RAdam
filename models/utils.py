"""Useful utilities."""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

# from fairseq.models.wav2vec import Wav2Vec2Model
from transformers import Wav2Vec2Model

# from pyannote.audio import Inference
# from pyannote.audio.models.embedding import XVector
from resemblyzer import VoiceEncoder


def adversarial_loss(scores, as_real=True):
    if as_real:
        return torch.mean((1 - scores) ** 2)
    return torch.mean(scores ** 2)


def discriminator_loss(fake_scores, real_scores):
    loss = adversarial_loss(fake_scores, as_real=False) + adversarial_loss(real_scores, as_real=True)
    return loss


def l1_loss(output, target, mask):
    return torch.masked_select(output - target, mask).abs().mean()


def mse_loss(output, target, mask):
    return (torch.masked_select(output - target, mask) ** 2).mean() ** 0.5


def mel_spec_loss(output, target, mask=None, alpha=0.05):
    if mask is None:
        mask = torch.full((output.shape[0], output.shape[2]), True)
    mask = mask.unsqueeze(1)

    if alpha == 0.:
        return l1_loss(output, target, mask)
    return l1_loss(output, target, mask) + alpha * mse_loss(output, target, mask)


def cosine_sim_loss(emb1, emb2):
    return (1. - F.cosine_similarity(emb1, emb2)).mean()


def load_pretrained_wav2vec(ckpt_path):
    """Load pretrained Wav2Vec model."""
    # ckpt = torch.load(ckpt_path)
    # model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
    # model.load_state_dict(ckpt["model"])
    # model.remove_pretraining_modules()
    # model.eval()

    def extract_features(self, wav, mask):
        # wav2vec has window of 400, so we pad to center windows
        wav = torch.nn.functional.pad(wav.unsqueeze(1), (200, 200), mode='reflect').squeeze(1)
        return [self(wav).last_hidden_state]

    Wav2Vec2Model.extract_features = extract_features # for same behaviour as fairseq.Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(ckpt_path).eval()
    return model


def load_pretrained_spk_emb(train=False, device='cpu', n_mels=80):
    """Load speaker embedding model"""
    # model = Inference('hbredin/SpeakerEmbedding-XVectorMFCC-VoxCeleb', device=device, window='sliding')from resemblyzer import VoiceEncoder, preprocess_wav
    # if train:
    #     model = XVector()
    #     model.sincnet = torch.nn.Conv1d(n_mels, model.frame1.input_dim, 3, 2)
    #     model = model.train().to(device)

    model = VoiceEncoder()
    return model


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = 1e-7,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
