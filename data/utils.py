"""Utilities for data manipulation."""

from typing import Union
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import lfilter

matplotlib.use("Agg")

def infinite_iterator(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)


def load_wav(
    audio_path: Union[str, Path], sample_rate: int, trim: bool = False
) -> np.ndarray:
    """Load and preprocess waveform."""
    wav = librosa.load(audio_path, sr=sample_rate)[0]
    wav = wav / (np.abs(wav).max() + 1e-6)
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:  # prevent empty slice
            wav = wav[start:end]

    return wav


def log_mel_spectrogram(
    x: np.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
    f_max: int
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    x = lfilter([1, -preemph], [1], x)
    magnitude = np.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec.T


def get_energy(wav, sr, n_fft=1280, hop_length=320, win_length=None, ref=1.0, min_db=-80.0):
  """
  Extract the loudness measurement of the signal.
  Feature is extracted using A-weighting of the signal frequencies.
  Args:
    wav          - waveform (numpy array)
    sr           - sampling rate
    n_fft        - number of points for fft
    hop_length   - stride of stft
    win_length   - size of window of stft
    ref          - reference for amplitude log-scale
    min_db       - floor for db difference
  Returns:
    loudness     - loudness of signal, shape (n_frames,) 
  """

  A_weighting = librosa.A_weighting(librosa.fft_frequencies(sr, n_fft=n_fft)+1e-6, min_db=min_db)
  weighting = 10 ** (A_weighting / 10)

  power_spec = abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) ** 2
  loudness = np.mean(power_spec * weighting[:, None], axis=0)
  loudness = librosa.power_to_db(loudness, ref=ref) # in db

  loudness = loudness.astype(np.float32)

  return (loudness - loudness.mean()) / loudness.std()

# best practice is to make f0 continuous and logarithmed
def convert_continuos_f0(f0):
  """CONVERT F0 TO CONTINUOUS F0
  Reference:
  https://github.com/bigpon/vcc20_baseline_cyclevae/blob/master/baseline/src/bin/feature_extract.py
  Args:
      f0 (ndarray): original f0 sequence with the shape (T)
  Return:
      (ndarray): continuous f0 with the shape (T)
  """
  # get uv information as binary
  uv = np.float32(f0 != 0)

  # get start and end of f0
  start_f0 = f0[f0 != 0][0]
  end_f0 = f0[f0 != 0][-1]

  # padding start and end of f0 sequence
  start_idx = np.where(f0 == start_f0)[0][0]
  end_idx = np.where(f0 == end_f0)[0][-1]
  f0[:start_idx] = start_f0
  f0[end_idx:] = end_f0

  # get non-zero frame index
  nz_frames = np.where(f0 != 0)[0]

  # perform linear interpolation
  f = scipy.interpolate.interp1d(nz_frames, f0[nz_frames])
  cont_f0 = f(np.arange(0, f0.shape[0]))

  return np.log(cont_f0)


def get_f0(wav, sr, hop_ms, f_min=50, f_max=800):
  """
  Extract f0 (1d-array of frame values) from wav (1d-array of point values).
  Args:
    wav    - waveform (numpy array)
    sr     - sampling rate
    hop_ms - stride (in milliseconds) for frames
    f_min  - f0 floor frequency
    f_max  - f0 ceil frequency
  Returns:
    f0     - interpolated main frequency, shape (n_frames,) 
  """
  if f_max is None:
    f_max = sr / 2

  _f0, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_ms, f0_floor=f_min, f0_ceil=f_max) # raw pitch extractor
  f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr)  # pitch refinement

  cont_f0 = convert_continuos_f0(f0).astype(np.float32)

  # norm to [-1, 1]
  mean = (np.log(f_max) + np.log(f_min)) / 2
  std = (np.log(f_max) - np.log(f_min)) / 2

  return (cont_f0 - mean) / std


def plot_mel(gt_mel, predicted_mel=None, filename="mel.png"):
    if predicted_mel is not None:
        fig, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 10))
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))

    axes[0][0].imshow(gt_mel.detach().cpu().numpy().T, origin="lower")
    axes[0][0].set_aspect(1, adjustable="box")
    axes[0][0].set_ylim(1.0, 80)
    axes[0][0].set_title("ground-truth mel-spectrogram", fontsize="medium")
    axes[0][0].tick_params(labelsize="x-small", left=False, labelleft=False)

    if predicted_mel is not None:
        axes[1][0].imshow(predicted_mel.detach().cpu().numpy(), origin="lower")
        axes[1][0].set_aspect(1.0, adjustable="box")
        axes[1][0].set_ylim(0, 80)
        axes[1][0].set_title("predicted mel-spectrogram", fontsize="medium")
        axes[1][0].tick_params(labelsize="x-small", left=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def get_mel_plot(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram.cpu().numpy(), aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_attn(attn, filename="attn.png"):
    fig, axes = plt.subplots(len(attn), 1, squeeze=False, figsize=(10, 10))

    for i, layer_attn in enumerate(attn):
        im = axes[i][0].imshow(attn[i][0].detach().cpu().numpy(), aspect='auto', origin="lower")
        axes[i][0].set_title("layer {}".format(i), fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small")
        axes[i][0].set_xlabel("target")
        axes[i][0].set_ylabel("source")
        fig.colorbar(im, ax=axes[i][0])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
