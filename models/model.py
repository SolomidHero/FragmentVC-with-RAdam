"""FragmentVC model architecture."""

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .convolutional_transformer import Smoother, Extractor


class FragmentVC(nn.Module):
    """
    FragmentVC uses Wav2Vec feature of the source speaker to query and attend
    on mel spectrogram of the target speaker.
    """

    def __init__(self, d_model=512, d_emb=None):
        super().__init__()

        self.unet = UnetBlock(d_model, d_emb=d_emb)

        self.smoothers = nn.TransformerEncoder(Smoother(d_model, 4, 1024), num_layers=3)

        self.mel_linear = nn.Linear(d_model, 80)

        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        ref_embs: Optional[Tensor] = None,
        refs_features: Optional[Tensor] = None,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
        pitch_target: Optional[Tensor] = None,
        energy_target: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            ref_embs: (batch, d_emb)
            refs: (batch, 80, ref_len)
            refs_features: (batch, ref_len, 768)
            ref_masks: (batch, ref_len)
        """

        # out: (src_len, batch, d_model)
        out, p_prediction, e_prediction, attns = self.unet(
            srcs, refs, ref_embs=ref_embs, refs_features=refs_features, src_masks=src_masks, ref_masks=ref_masks,
            pitch_target=pitch_target, energy_target=energy_target
        )

        # out: (src_len, batch, d_model)
        out = self.smoothers(out, src_key_padding_mask=src_masks)

        # out: (src_len, batch, 80)
        out = self.mel_linear(out)

        # out: (batch, 80, src_len)
        out = out.transpose(1, 0).transpose(2, 1)
        refined = self.post_net(out)
        out = out + refined

        # out: (batch, 80, src_len)
        return out, p_prediction, e_prediction, [attn1, attn2, attn3]


class UnetBlock(nn.Module):
    """Hierarchically attend on references."""

    def __init__(self, d_model: int, d_feat: int = 768, d_emb: int = None):
        super(UnetBlock, self).__init__()

        self.conv1 = nn.Conv1d(80, d_model, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")

        self.prenet = nn.Linear(d_feat, d_model)
        nn.init.orthogonal_(self.prenet.weight)
        self.features_prenet = nn.Linear(d_feat, d_model)
        nn.init.orthogonal_(self.prenet.weight)

        self.use_emb = d_emb is not None
        if self.use_emb:
            self.emb_prenet = nn.Sequential(
                nn.Linear(d_emb, 256),
                nn.ReLU(),
                nn.Linear(256, d_model),
            )

        self.pitch_predictor = VariancePredictor(d_model)
        self.energy_predictor = VariancePredictor(d_model)

        self.pitch_bins = torch.linspace(-1.0, 1.0, 256 - 1)
        self.energy_bins = torch.linspace(-2.0, 10.0, 256 - 1)

        self.pitch_embedding = nn.Embedding(256, d_model)
        self.energy_embedding = nn.Embedding(256, d_model)

        self.extractor1 = Extractor(d_model, 4, 1024, no_residual=False)
        self.extractor2 = Extractor(d_model, 4, 1024)
        self.extractor3 = Extractor(d_model, 4, 1024)

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        ref_embs: Optional[Tensor] = None,
        refs_features: Optional[Tensor] = None,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
        pitch_target: Optional[Tensor] = None,
        energy_target: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_embs: (batch, d_emb)
            refs_features: (batch, ref_len, 768)
            ref_masks: (batch, ref_len)
        """


        assert int(ref_embs is not None) == int(self.use_emb)

        # tgt: (batch, tgt_len, d_model)
        tgt = self.prenet(srcs)
        refs_features = None if refs_features is None else self.features_prenet(refs_features).transpose(0, 1)

        if self.use_emb:
            # ref_emb: (batch, d_model)
            ref_embs = self.emb_prenet(ref_embs)
            tgt = tgt + ref_embs.unsqueeze(1)

        # pitch_prediction: (batch, tgt_len)
        pitch_prediction = self.pitch_predictor(tgt, src_masks)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_prediction, self.pitch_bins))

        # energy_prediction: (batch, tgt_len)
        energy_prediction = self.energy_predictor(tgt, src_masks)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_target, self.energy_bins))
        else:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_prediction, self.energy_bins))

        # tgt: (batch, tgt_len, d_model)
        tgt = tgt + pitch_embedding + energy_embedding

        # tgt: (tgt_len, batch, d_model)
        tgt = tgt.transpose(0, 1)

        # ref*: (batch, d_model, mel_len)
        ref1 = self.conv1(refs)
        ref2 = self.conv2(F.relu(ref1))
        ref3 = self.conv3(F.relu(ref2))

        # out*: (tgt_len, batch, d_model)
        out, attn1 = self.extractor1(
            tgt,
            ref3.transpose(1, 2).transpose(0, 1),
            memory_features=refs_features,
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )
        out, attn2 = self.extractor2(
            out,
            ref2.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )
        out, attn3 = self.extractor3(
            out,
            ref1.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )

        # out: (tgt_len, batch, d_model)
        return out, pitch_prediction, energy_prediction, [attn1, attn2, attn3]


class VariancePredictor(nn.Module):
    """ Pitch and Energy Predictor """

    def __init__(self, d_model: int, d_hidden: int = 256, dropout: float=0.5):
        super(VariancePredictor, self).__init__()


        self.conv_layer = nn.Sequential(
            nn.Conv1d(d_model, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(self.dropout),
            nn.Conv1d(d_hidden, 1, kernel_size=1),
        )


    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = out.squeeze(-2)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Discriminator(nn.Module):
    """Score the realness of melspectrogram."""
    def __init__(self):
        super().__init__()

        self.inputConvLayer = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.SiLU()
        )

        # DownSample Layer
        self.down1 = self.downSample(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down2 = self.downSample(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.down3 = self.downSample(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.down4 = self.downSample(in_channels=1024, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))

        # Conv Layer
        self.outputConvLayer = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.SiLU()
        )

    def forward(self, input):
        # input has shape (batch_size, num_features, time)
        # discriminator requires shape (batchSize, 1, num_features, time)
        x = self.inputConvLayer(input.unsqueeze(1))

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        # x = self.down4(x)

        output = self.outputConvLayer(x)
        return output



class ConditionalDiscriminator(nn.Module):
    """Score match of melspectrogram and embedding."""
    def __init__(self, in_c=80, cond_dim=512):
        super().__init__()

        self.inputConvLayer = nn.Sequential(
          nn.Conv1d(in_channels=in_c, out_channels=128, kernel_size=3, stride=1, padding=0),
          nn.SiLU()
        )

        # DownSample Layer
        self.down1 = self.convBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.down2 = self.convBlock(in_channels=256, out_channels=512, kernel_size=5, stride=3, padding=2)
        self.down3 = self.convBlock(in_channels=512, out_channels=1024, kernel_size=11, stride=3, padding=5)
        # self.down4 = self.convBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0)

        # Conv Layer
        self.out1 = self.convBlock(in_channels=1024 + cond_dim, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.out2 = nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0)

    def convBlock(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation
            ),
            nn.InstanceNorm1d(num_features=out_channels, affine=True),
            nn.SiLU()
        )

    def forward(self, input, cond):
        # input has shape (batch_size, num_features, time)
        # cond  has shape (batch_size, emb_dim)
        # discriminator requires shape (batchSize, num_features, time)
        x = self.inputConvLayer(input)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        cond = torch.repeat_interleave(cond, x.shape[2], dim=1).reshape(cond.shape[0], cond.shape[1], x.shape[2])
        x = torch.cat([x, cond], dim=1)
        x = self.out1(x)
        output = self.out2(x)
        return output
