import torch
from torch import nn


class ConditioningAugmentationNetwork(nn.Module):
    def __init__(
        self,
        speech_emb_dim: int = 1024,
        gan_emb_dim: int = 128,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.speech_dim = speech_emb_dim
        self.gan_emb_dim = gan_emb_dim
        self.fc = nn.Linear(self.speech_dim, self.gan_emb_dim * 4)
        # multiply by 2 if leaky relu
        # multiply by 4 if GLU
        # self.act = nn.LeakyReLU(negative_slope)
        self.act = nn.GLU(dim=1)

    def forward(self, x):
        """
        x: speech embedding vector
        """
        x = self.act(self.fc(x))
        mu = x[:, : self.gan_emb_dim]
        logvar = x[:, self.gan_emb_dim :]
        std = torch.exp(logvar.mul(0.5))
        z = torch.randn(std.size()).to(x.device)
        z_ca = mu + std * z
        return z_ca, mu, logvar


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, size=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size

    def forward(self, x):
        return self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, size=self.size
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_seq = Interpolate(scale_factor=2, mode="nearest")
        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        x = self.first_seq(x)
        x = self.seq(x)
        return x


class Block(nn.Module):
    def __init__(self, desc):
        super().__init__()
        self.desc = desc

    def forward(self, x):
        return x


class InitStateGenerator(nn.Module):
    def __init__(self, in_dim: int, gen_dim: int):
        super().__init__()
        # in_dim = z_dim + gan_emb_dim
        self.in_dim = in_dim
        self.gen_dim = gen_dim
        self.input_projection = nn.Sequential(
            nn.Linear(self.in_dim, self.gen_dim * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.gen_dim * 4 * 4 * 2),
            nn.GLU(dim=1),
        )
        self.seq_upsample = nn.Sequential(
            UpBlock(gen_dim, gen_dim // 2),
            UpBlock(gen_dim // 2, gen_dim // 4),
            UpBlock(gen_dim // 4, gen_dim // 8),
            UpBlock(gen_dim // 8, gen_dim // 16),
        )

    def forward(self, z_code, c_code):
        inp = torch.cat((z_code, c_code), 1)
        out = self.input_projection(inp)
        out = out.view(-1, self.gen_dim, 4, 4)
        # up from 4x4 -> 64x64
        out = self.seq_upsample(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
            nn.GLU(dim=1),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return self.block(x) + x


class NextStageGenerator(nn.Module):
    def __init__(self, gan_emb_dim: int, gen_dim: int):
        super().__init__()
        self.gan_emb_dim = gan_emb_dim
        self.gen_dim = gen_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(
                gan_emb_dim + gen_dim,
                gen_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(gen_dim * 2),
            nn.GLU(dim=1),
        )
        self.residual = nn.Sequential(ResBlock(gen_dim), ResBlock(gen_dim))
        self.upsample = UpBlock(gen_dim * 2, gen_dim // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.gan_emb_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)

        concat_code = torch.cat((c_code, h_code), 1)

        out = self.joint_conv(concat_code)
        out = self.residual(out)
        out = torch.cat((out, h_code), 1)
        out = self.upsample(out)
        return out


class ImageGenerator(nn.Module):
    def __init__(self, gen_dim: int):
        super().__init__()
        self.img = nn.Sequential(
            nn.Conv2d(gen_dim, 3, kernel_size=3, stride=1, padding=1, bias=False,),
            nn.Tanh(),
        )

    def forward(self, h_code):
        return self.img(h_code)


class DenselyStackedGenerator(nn.Module):
    def __init__(
        self, latent_space_dim: int, speech_emb_dim: int, gen_dim: int, gan_emb_dim: int
    ):
        super().__init__()
        inp_dim = latent_space_dim + gan_emb_dim

        self.conditioning_augmentation = ConditioningAugmentationNetwork(
            speech_emb_dim=speech_emb_dim, gan_emb_dim=gan_emb_dim
        )

        self.F0 = InitStateGenerator(in_dim=inp_dim, gen_dim=gen_dim * 16)
        self.G0 = ImageGenerator(gen_dim=gen_dim)

        self.F1 = NextStageGenerator(gan_emb_dim=gan_emb_dim, gen_dim=gen_dim)
        self.G1 = ImageGenerator(gen_dim=gen_dim // 2)

        self.F2 = NextStageGenerator(gan_emb_dim=gan_emb_dim, gen_dim=gen_dim // 2)
        self.G2 = ImageGenerator(gen_dim=gen_dim // 4)

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, z_code, speech_emb):
        c_code, mu, logvar = self.conditioning_augmentation(speech_emb)

        h0 = self.F0(z_code, c_code)
        h1 = self.F1(h0, c_code)
        h2 = self.F2(h1, c_code)

        fake_imgs = {64: self.G0(h0), 128: self.G1(h1), 256: self.G2(h2)}

        return fake_imgs, mu, logvar
