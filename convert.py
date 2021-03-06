#!/usr/bin/env python3
"""Convert using one source utterance and multiple target utterances."""

import warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
import soundfile as sf
from jsonargparse import ArgumentParser, ActionConfigFile

import sox

from data import load_wav, log_mel_spectrogram, plot_mel, plot_attn
from models import load_pretrained_wav2vec


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("source_path", type=str)
    parser.add_argument("target_paths", type=str, nargs="+")
    parser.add_argument("-w", "--wav2vec_path", type=str, required=True)
    parser.add_argument("-c", "--ckpt_path", type=str, required=True)
    parser.add_argument("-v", "--vocoder_path", type=str, default="vocoder.pt")
    parser.add_argument("-o", "--output_path", type=str, default="output.wav")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--hop_len", type=int, default=320)
    parser.add_argument("--win_len", type=int, default=1280)
    parser.add_argument("--n_fft", type=int, default=1280)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--f_min", type=int, default=80)
    parser.add_argument("--f_max", type=int, default=None)
    parser.add_argument("--mel_only", action='store_true')
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--use_target_features", action='store_true')
    parser.add_argument("--audio_config", action=ActionConfigFile)

    return vars(parser.parse_args())


def main(
    source_path,
    target_paths,
    wav2vec_path,
    ckpt_path,
    vocoder_path,
    output_path,
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
    use_target_features,
    **kwargs,
):
    """Main function."""

    begin_time = step_moment = datetime.now()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
    print("[INFO] Wav2Vec is loaded from", wav2vec_path)

    model = torch.jit.load(ckpt_path).to(device).eval()
    print("[INFO] FragmentVC is loaded from", ckpt_path)

    if not mel_only:
        vocoder = torch.jit.load(vocoder_path).to(device).eval()
        print("[INFO] Vocoder is loaded from", vocoder_path)

    elaspe_time = datetime.now() - step_moment
    step_moment = datetime.now()
    print("[INFO] elasped time", elaspe_time.total_seconds())

    tfm = sox.Transformer()
    tfm.vad(location=1)
    tfm.vad(location=-1)

    src_wav = load_wav(source_path, sample_rate)
    src_wav = deepcopy(tfm.build_array(input_array=src_wav, sample_rate_in=sample_rate))
    src_wav = torch.FloatTensor(src_wav).unsqueeze(0).to(device)
    print("[INFO] source waveform shape:", src_wav.shape)

    tgt_mels = []
    tgt_feats = []
    for tgt_path in target_paths:
        tgt_wav = load_wav(tgt_path, sample_rate)
        tgt_wav = tfm.build_array(input_array=tgt_wav, sample_rate_in=sample_rate)
        tgt_wav = deepcopy(tgt_wav)
        tgt_mel = log_mel_spectrogram(
            tgt_wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min, f_max
        )
        tgt_mels.append(tgt_mel)
        if use_target_features:
            with torch.no_grad():
                tgt_feats.append(wav2vec.extract_features(torch.from_numpy(tgt_wav[None]).to(device), None)[0])


    if use_target_features:
        tgt_feat = torch.cat(tgt_feats, dim=1)
    else:
        tgt_feat = None

    tgt_mel = np.concatenate(tgt_mels, axis=0)
    tgt_mel = torch.FloatTensor(tgt_mel.T).unsqueeze(0).to(device)
    print("[INFO] target spectrograms shape:", tgt_mel.shape)

    with torch.no_grad():
        src_feat = wav2vec.extract_features(src_wav, None)[0]
        print("[INFO] source Wav2Vec feature shape:", src_feat.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

        out_mel, attns = model(src_feat, tgt_mel)
        out_mel = out_mel.transpose(1, 2).squeeze(0)
        print("[INFO] converted spectrogram shape:", out_mel.shape)

        elaspe_time = datetime.now() - step_moment
        step_moment = datetime.now()
        print("[INFO] elasped time", elaspe_time.total_seconds())

        if not mel_only:
            out_wav = vocoder.generate([out_mel])[0]
            out_wav = out_wav.cpu().numpy()
            print("[INFO] generated waveform shape:", out_wav.shape)

            elaspe_time = datetime.now() - step_moment
            step_moment = datetime.now()
            print("[INFO] elasped time", elaspe_time.total_seconds())

    wav_path = Path(output_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if mel_only:
        mel_path = wav_path.with_suffix(".npy")
        np.save(mel_path, out_mel.cpu().numpy())
        print("[INFO] mel-spectrogram .npy is saved to", mel_path)
    else:
        sf.write(wav_path, out_wav, sample_rate)
        print("[INFO] generated waveform is saved to", wav_path)

    if plot:
        mel_path = wav_path.with_suffix(".mel.png")
        plot_mel(out_mel, filename=mel_path)
        print("[INFO] mel-spectrogram plot is saved to", mel_path)

        attn_path = wav_path.with_suffix(".attn.png")
        plot_attn(attns, filename=attn_path)
        print("[INFO] attention plot is saved to", attn_path)

    elaspe_time = datetime.now() - begin_time
    print("[INFO] Overall elasped time", elaspe_time.total_seconds())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(**parse_args())
