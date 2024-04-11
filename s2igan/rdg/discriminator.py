import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownScale16TimesBlock(nn.Module):
    def __init__(self, disc_dim: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, disc_dim, kernel_size=4, stride=2, padding=1, bias=False,),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                disc_dim, disc_dim * 2, kernel_size=4, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(disc_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                disc_dim * 2,
                disc_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                disc_dim * 4,
                disc_dim * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        return self.seq(img)


class DiscriminatorFor64By64(nn.Module):
    def __init__(self, disc_dim: int, gan_emb_dim: int):
        super().__init__()
        self.disc_dim = disc_dim
        self.gan_emb_dim = gan_emb_dim
        self.down_scale = DownScale16TimesBlock(disc_dim)
        self.joint_conv = nn.Sequential(
            nn.Conv2d(
                disc_dim * 8 + gan_emb_dim,
                disc_dim * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logits = nn.Sequential(
            nn.Conv2d(disc_dim * 8, 1, kernel_size=4, stride=4), nn.Sigmoid()
        )
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(disc_dim * 8, 1, kernel_size=4, stride=1), nn.Sigmoid()
        )

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x_var, c_code):
        x_code = self.down_scale(x_var)
        c_code = c_code.view(-1, self.gan_emb_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        code = torch.cat((c_code, x_code), 1)
        code = self.joint_conv(code)

        output = self.logits(code)
        uncond_output = self.uncond_logits(x_code)

        return {"cond": output.view(-1), "uncond": uncond_output.view(-1)}


class DiscriminatorFor128By128(nn.Module):
    def __init__(self, disc_dim: int, gan_emb_dim: int):
        super().__init__()
        self.disc_dim = disc_dim
        self.gan_emb_dim = gan_emb_dim
        self.down_scale = nn.Sequential(
            DownScale16TimesBlock(disc_dim),
            DownBlock(disc_dim * 8, disc_dim * 16),
            nn.Sequential(
                nn.Conv2d(
                    disc_dim * 16,
                    disc_dim * 8,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(disc_dim * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        )
        self.joint_conv = nn.Sequential(
            nn.Conv2d(
                disc_dim * 8 + gan_emb_dim,
                disc_dim * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logits = nn.Sequential(
            nn.Conv2d(disc_dim * 8, 1, kernel_size=4, stride=4), nn.Sigmoid()
        )
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(disc_dim * 8, 1, kernel_size=4, stride=1), nn.Sigmoid()
        )

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x_var, c_code):
        x_code = self.down_scale(x_var)

        c_code = c_code.view(-1, self.gan_emb_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        code = torch.cat((c_code, x_code), 1)
        code = self.joint_conv(code)

        output = self.logits(code)
        uncond_output = self.uncond_logits(x_code)

        return {"cond": output.view(-1), "uncond": uncond_output.view(-1)}


class DiscriminatorFor256By256(nn.Module):
    def __init__(self, disc_dim: int, gan_emb_dim: int):
        super().__init__()
        self.disc_dim = disc_dim
        self.gan_emb_dim = gan_emb_dim
        self.down_scale = nn.Sequential(
            DownScale16TimesBlock(disc_dim),
            DownBlock(disc_dim * 8, disc_dim * 16),
            DownBlock(disc_dim * 16, disc_dim * 32),
            nn.Sequential(
                nn.Conv2d(
                    disc_dim * 32,
                    disc_dim * 16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(disc_dim * 16),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(
                    disc_dim * 16,
                    disc_dim * 8,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(disc_dim * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        )
        self.joint_conv = nn.Sequential(
            nn.Conv2d(
                disc_dim * 8 + gan_emb_dim,
                disc_dim * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logits = nn.Sequential(
            nn.Conv2d(disc_dim * 8, 1, kernel_size=4, stride=4), nn.Sigmoid()
        )
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(disc_dim * 8, 1, kernel_size=4, stride=1), nn.Sigmoid()
        )

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x_var, c_code):
        x_code = self.down_scale(x_var)

        c_code = c_code.view(-1, self.gan_emb_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        code = torch.cat((c_code, x_code), 1)
        code = self.joint_conv(code)

        output = self.logits(code)
        uncond_output = self.uncond_logits(x_code)

        return {"cond": output.view(-1), "uncond": uncond_output.view(-1)}
