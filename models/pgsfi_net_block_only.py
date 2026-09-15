from models.unet_parts import *
from models.spatial_freq_domain import (
    MultiBranchSpatialFreq,
    VesselPriorExtractor,
    Plain_Four_SAFM_Block,
    DecoderParallelBlock
)


class PGSFINetBlockOnly(nn.Module):
    """消融：完整版，但 cross-gating 只保留 block gating（关闭 grid gating）"""

    def __init__(self, n_channels, n_classes, downsize_nb_filters_factor=4, dropout_rate=0.1):
        super(PGSFINetBlockOnly, self).__init__()
        f = downsize_nb_filters_factor

        self.inc = inconv(n_channels, 64 // f)
        self.down1 = down(64 // f, 128 // f)
        self.down2 = down(128 // f, 256 // f)
        self.down3 = down(256 // f, 512 // f)
        self.down4 = down(512 // f, 512 // f)

        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.four_block3 = Plain_Four_SAFM_Block(512 // f)
        self.four_block4 = Plain_Four_SAFM_Block(512 // f)

        self.spatial_freq_block = MultiBranchSpatialFreq(
            in_dim=512 // f,
            num_blocks=1,
            grid_size=(1, 1),
            block_size=(3, 3),
            gate_type='block_only'
        )

        self.prior_extractor = VesselPriorExtractor(in_channels=1, out_channels=64 // f)

        self.prior_p5 = nn.Sequential(
            nn.MaxPool2d(16),
            nn.Conv2d(64 // f, 512 // f, 3, 1, 1),
            nn.BatchNorm2d(512 // f),
            nn.ReLU(inplace=True)
        )
        self.prior_up1 = nn.Sequential(
            nn.MaxPool2d(8),
            nn.Conv2d(64 // f, 256 // f, 3, 1, 1),
            nn.BatchNorm2d(256 // f),
            nn.ReLU(inplace=True)
        )
        self.prior_up2 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(64 // f, 128 // f, 3, 1, 1),
            nn.BatchNorm2d(128 // f),
            nn.ReLU(inplace=True)
        )
        self.prior_up3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64 // f, 64 // f, 3, 1, 1),
            nn.BatchNorm2d(64 // f),
            nn.ReLU(inplace=True)
        )
        self.prior_up4 = nn.Sequential(
            nn.Conv2d(64 // f, 64 // f, 3, 1, 1),
            nn.BatchNorm2d(64 // f),
            nn.ReLU(inplace=True)
        )

        self.up1 = up(1024 // f, 256 // f)
        self.up2 = up(512 // f, 128 // f)
        self.up3 = up(256 // f, 64 // f)
        self.up4 = up(128 // f, 64 // f)

        self.decoder_block1 = DecoderParallelBlock(256 // f)
        self.decoder_block2 = DecoderParallelBlock(128 // f)
        self.decoder_block3 = DecoderParallelBlock(64 // f)
        self.decoder_block4 = DecoderParallelBlock(64 // f)

        self.outc = nn.Conv2d(64 // f + 1, n_classes, 1)

    def forward(self, inp):
        prior = self.prior_extractor(inp)

        p5 = self.prior_p5(prior)
        p_up1 = self.prior_up1(prior)
        p_up2 = self.prior_up2(prior)
        p_up3 = self.prior_up3(prior)
        p_up4 = self.prior_up4(prior)

        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)
        x4 = self.four_block3(x4)

        x5 = self.down4(x4)
        x5 = self.four_block4(x5)

        x5 = x5 + p5
        x5 = self.spatial_freq_block(x5)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.decoder_block1(x)
        x = x + p_up1

        x = self.up2(x, x3)
        x = self.decoder_block2(x)
        x = x + p_up2

        x = self.up3(x, x2)
        x = self.decoder_block3(x)
        x = x + p_up3

        x = self.up4(x, x1)
        x = self.decoder_block4(x)
        x = x + p_up4

        x = torch.cat([inp, x], dim=1)
        x = self.outc(x)
        return torch.sigmoid(x)
