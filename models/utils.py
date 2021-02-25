"""Useful utilities."""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# from fairseq.models.wav2vec import Wav2Vec2Model
from transformers import Wav2Vec2Model


def load_pretrained_wav2vec(ckpt_path):
    """Load pretrained Wav2Vec model."""
    # ckpt = torch.load(ckpt_path)
    # model = Wav2Vec2Model.build_model(ckpt["args"], task=None)
    # model.load_state_dict(ckpt["model"])
    # model.remove_pretraining_modules()
    # model.eval()

    def extract_features(self, wav, mask):
        # wav2vec has window of 400, so we pad to center windows
        return [self(torch.nn.functional.pad(wav, (200, 200), mode='reflect')).last_hidden_state]

    Wav2Vec2Model.extract_features = extract_features # for same behaviour as fairseq.Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(ckpt_path)
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
