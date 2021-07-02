#!/usr/bin/env python3
"""Convert multiple pairs."""

import warnings
from pathlib import Path
from functools import partial
from torch.multiprocessing import Pool, cpu_count

import yaml
import torch
import numpy as np
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile
from resemblyzer import preprocess_wav

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from models import load_pretrained_wav2vec, load_pretrained_spk_emb


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("info_path", type=str)
    parser.add_argument("output_dir", type=str, default=".")
    parser.add_argument("-c", "--ckpt_path", default="checkpoints/fragmentvc.pt")
    parser.add_argument("-w", "--wav2vec_path", default="checkpoints/wav2vec_small.pt")
    parser.add_argument("-v", "--vocoder_path", default="checkpoints/vocoder.pt")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop_len", type=int, default=320)
    parser.add_argument("--win_len", type=int, default=1280)
    parser.add_argument("--n_fft", type=int, default=1280)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--f_min", type=int, default=80)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--mel_only", action='store_true')
    parser.add_argument("--trim", action='store_true')
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--use_spk_emb", action='store_true')
    parser.add_argument("--use_target_features", action='store_true')
    parser.add_argument("--audio_config", action=ActionConfigFile)

    return vars(parser.parse_args())


def main(
    info_path,
    output_dir,
    ckpt_path,
    wav2vec_path,
    vocoder_path,
    sample_rate,
    preemph,
    hop_len,
    win_len,
    n_fft,
    n_mels,
    f_min,
    f_max,
    mel_only,
    plot,
    trim,
    use_spk_emb,
    use_target_features,
    **kwargs,
):
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
    print("[INFO] Wav2Vec is loaded from", wav2vec_path)

    model = torch.jit.load(ckpt_path).to(device).eval()
    print("[INFO] FragmentVC is loaded from", ckpt_path)

    if not mel_only:
        vocoder = torch.jit.load(vocoder_path).to(device).eval()
        print("[INFO] Vocoder is loaded from", vocoder_path)

    path2wav = partial(load_wav, sample_rate=sample_rate)
    wav2mel = partial(
        log_mel_spectrogram,
        preemph=preemph,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_len,
        win_length=win_len,
        f_min=f_min,
        f_max=f_max,
    )
    wav2emb = load_pretrained_spk_emb(train=False, device=device)

    with open(info_path) as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)

    out_mels = []
    attns = []
    pair_names = []

    for pair_name, pair in infos.items():
        if isinstance(pair["source"], str):
            pair["source"] = { pair_name: pair["source"] }

        with Pool(cpu_count()) as pool:
            tgt_wavs = pool.map(path2wav, pair["target"])
            tgt_mels = pool.map(wav2mel, tgt_wavs)

        if use_target_features:
            with torch.no_grad():
                tgt_feats = list(map(
                    lambda x: wav2vec.extract_features(torch.from_numpy(x[None]).to(device), None)[0],
                    tgt_wavs
                ))
            tgt_feat = torch.cat(tgt_feats, dim=1)
        else:
            tgt_feat = None

        tgt_mel = np.concatenate(tgt_mels, axis=0)
        tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(device)

        if use_spk_emb:
            tgt_emb = wav2emb.embed_utterance(preprocess_wav(np.concatenate(tgt_wavs), sample_rate))
            tgt_emb = torch.from_numpy(tgt_emb).to(device).unsqueeze(0)
        else:
            tgt_emb = None


        pair_names.extend(pair["source"].keys())
        for cur_pair_name, source in pair["source"].items():
            src_wav = load_wav(source, sample_rate, trim=trim)
            src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(device)

            with torch.no_grad():
                src_feat = wav2vec.extract_features(src_wav, None)[0]

                out_mel, _, _, attn = model(src_feat, tgt_mel, ref_embs=tgt_emb)
                out_mel = out_mel.transpose(1, 2).squeeze(0)

                out_mels.append(out_mel.cpu() if mel_only else out_mel)
                if plot:
                    attns.append([x.cpu() for x in attn])

            print(f"[INFO] Pair {cur_pair_name} converted")


    if not mel_only:
        print("[INFO] Generating waveforms...")
        with torch.no_grad():
            out_wavs = vocoder.generate(out_mels)
        print("[INFO] Waveforms generated")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        print("[INFO] Generating plots...")
        for pair_name, out_mel, attn in zip(
            pair_names, out_mels, attns
        ):
            out_path = Path(out_dir, pair_name)
            plot_mel(out_mel, filename=out_path.with_suffix(".mel.png"))
            plot_attn(attn, filename=out_path.with_suffix(".attn.png"))

    print("[INFO] Saving results...")
    if not mel_only:
        for pair_name, out_mel, out_wav in zip(
            pair_names, out_mels, out_wavs
        ):
            out_path = Path(out_dir, pair_name)
            sf.write(out_path.with_suffix(".wav"), out_wav.cpu().numpy(), sample_rate)
    else:
        for pair_name, out_mel in zip(
            pair_names, out_mels
        ):
            out_path = Path(out_dir, pair_name)
            np.save(out_path.with_suffix(".npy"), out_mel)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
