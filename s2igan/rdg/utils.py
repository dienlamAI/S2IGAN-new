import torch
from tqdm import tqdm
import random
import wandb
from torchvision import transforms as T


def get_transform(img_dim):
    return T.Compose([T.Resize(img_dim), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


Resizer = {64: get_transform(64), 128: get_transform(128), 256: get_transform(256)}


def update_D(
    models, batch, optimizers, schedulers, criterions, specific_params, device
):
    origin_real_img, origin_similar_img, origin_wrong_img, spec, spec_len, raw_audio = batch

    real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
    for img_dim in specific_params.img_dims:
        real_imgs[img_dim] = Resizer[img_dim](origin_real_img)
        wrong_imgs[img_dim] = Resizer[img_dim](origin_wrong_img)
        similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

    for key in optimizers.keys():
        optimizers[key].zero_grad()

    bs = origin_real_img.size(0)

    Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
    A = models["sed"](spec, spec_len)

    fake_imgs, mu, logvar = models["gen"](Z, A)

    zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
    one_labels = torch.ones(bs, device=device, dtype=torch.float)
    two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

    D_loss = 0
    for img_dim in specific_params.img_dims:
        optimizers[img_dim].zero_grad()

        real_img = real_imgs[img_dim]
        wrong_img = wrong_imgs[img_dim]

        real_out = models["disc"][img_dim](real_img, mu.detach())
        wrong_out = models["disc"][img_dim](wrong_img, mu.detach())
        fake_out = models["disc"][img_dim](fake_imgs[img_dim].detach(), mu.detach())

        loss_real_cond = criterions["bce"](real_out["cond"], one_labels)
        loss_real_uncond = criterions["bce"](real_out["uncond"], one_labels)
        # ---
        loss_wrong_cond = criterions["bce"](wrong_out["cond"], zero_labels)
        loss_wrong_uncond = criterions["bce"](wrong_out["uncond"], one_labels)
        # ---
        loss_fake_cond = criterions["bce"](fake_out["cond"], zero_labels)
        loss_fake_uncond = criterions["bce"](fake_out["uncond"], zero_labels)

        curr_D_loss = (
            loss_real_cond
            + loss_real_uncond
            + loss_fake_cond
            + loss_fake_uncond
            + loss_wrong_cond
            + loss_wrong_uncond
        )
        curr_D_loss.backward()
        optimizers[img_dim].step()
        schedulers[img_dim].step()

        D_loss += curr_D_loss.detach().item()

    return D_loss


def update_RS(
    models, batch, optimizers, schedulers, criterions, specific_params, device
):
    origin_real_img, origin_similar_img, origin_wrong_img, spec, spec_len, raw_audio = batch

    real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
    for img_dim in specific_params.img_dims:
        real_imgs[img_dim] = Resizer[img_dim](origin_real_img)
        wrong_imgs[img_dim] = Resizer[img_dim](origin_wrong_img)
        similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

    optimizers.zero_grad()

    bs = origin_real_img.size(0)

    Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
    A = models["sed"](spec, spec_len)

    fake_imgs, mu, logvar = models["gen"](Z, A)

    zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
    one_labels = torch.ones(bs, device=device, dtype=torch.float)
    two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

    real_img = Resizer[256](origin_real_img)
    similar_img = Resizer[256](origin_similar_img)
    wrong_img = Resizer[256](origin_wrong_img)

    real_feat = models["ied"](real_img)
    similar_feat = models["ied"](similar_img)
    fake_feat = models["ied"](fake_imgs[256].detach())
    wrong_feat = models["ied"](wrong_img)

    R1 = models["rs"](similar_feat.detach(), real_feat.detach())
    R2 = models["rs"](wrong_feat.detach(), real_feat.detach())
    R3 = models["rs"](real_feat.detach(), real_feat.detach())
    R_GT_FI = models["rs"](fake_feat.detach(), real_feat.detach())

    RS_loss = criterions["rs"](R1, R2, R3, R_GT_FI, zero_labels, one_labels, two_labels)

    RS_loss.backward()
    optimizers.step()
    schedulers.step()

    return RS_loss.detach().item()


def update_G(
    models, batch, optimizers, schedulers, criterions, specific_params, device
):
    origin_real_img, origin_similar_img, origin_wrong_img, spec, spec_len, raw_audio = batch

    real_imgs, wrong_imgs, similar_imgs = {}, {}, {}
    for img_dim in specific_params.img_dims:
        real_imgs[img_dim] = Resizer[img_dim](origin_real_img)
        wrong_imgs[img_dim] = Resizer[img_dim](origin_wrong_img)
        similar_imgs[img_dim] = Resizer[img_dim](origin_similar_img)

    optimizers.zero_grad()

    bs = origin_real_img.size(0)

    Z = torch.randn(bs, specific_params.latent_space_dim, device=device)
    A = models["sed"](spec, spec_len)

    fake_imgs, mu, logvar = models["gen"](Z, A)

    zero_labels = torch.zeros(bs, device=device, dtype=torch.float)
    one_labels = torch.ones(bs, device=device, dtype=torch.float)
    two_labels = torch.zeros(bs, device=device, dtype=torch.float) + 2

    G_loss = 0
    for img_dim in specific_params.img_dims:

        fake_out = models["disc"][img_dim](fake_imgs[img_dim], mu)
        cond_loss = criterions["bce"](fake_out["cond"], one_labels)
        uncond_loss = criterions["bce"](fake_out["uncond"], one_labels)

        wandb.log({f"train/cond_loss_{img_dim}": cond_loss.item()})
        wandb.log({f"train/uncond_loss_{img_dim}": uncond_loss.item()})

        G_loss += cond_loss + uncond_loss

        real_feat = models["ied"](real_imgs[img_dim])
        fake_feat = models["ied"](fake_imgs[img_dim])
        rs_out = models["rs"](real_feat, fake_feat)

        G_loss += criterions["ce"](rs_out, one_labels.long())

    # real_img = Resizer[256](origin_real_img)
    # similar_img = Resizer[256](origin_similar_img)
    # wrong_img = Resizer[256](origin_wrong_img)

    # real_feat = models["ied"](real_img)
    # similar_feat = models["ied"](similar_img)
    # fake_feat = models["ied"](fake_imgs[256])
    # wrong_feat = models["ied"](wrong_img)

    # R1 = models["rs"](similar_feat, real_feat)
    # R2 = models["rs"](wrong_feat, real_feat)
    # R3 = models["rs"](real_feat, real_feat)
    # R_GT_FI = models["rs"](fake_feat, real_feat)

    # RS_loss = criterions["rs"](R1, R2, R3, R_GT_FI, zero_labels, one_labels, two_labels)

    KL_loss = criterions["kl"](mu, logvar) * specific_params.kl_loss_coef
    # G_loss += RS_loss
    G_loss += KL_loss
    G_loss.backward()
    optimizers.step()
    schedulers.step()

    i = random.randint(0, origin_real_img.size(0) - 1)
    audio_path, sr = raw_audio[i]

    image_64 = torch.cat((fake_imgs[64][i:i+1], real_imgs[64][i:i+1]), 0) * 0.5 + 0.5
    image_128 = torch.cat((fake_imgs[128][i:i+1], real_imgs[128][i:i+1]), 0) * 0.5 + 0.5
    image_256 = torch.cat((fake_imgs[256][i:i+1], real_imgs[256][i:i+1]), 0) * 0.5 + 0.5

    wandb.log({"train/image_64": wandb.Image(image_64)})
    wandb.log({"train/image_128": wandb.Image(image_128)})
    wandb.log({"train/image_256": wandb.Image(image_256)})
    wandb.log({"train/speech_description": wandb.Audio(audio_path, sample_rate=sr)})

    return G_loss.detach().item(), KL_loss.detach().item()


def rdg_train_epoch(
    models,
    dataloader,
    optimizers,
    schedulers,
    criterions,
    specific_params,
    device,
    epoch,
    log_wandb,
):
    size = len(dataloader)
    pbar = tqdm(dataloader, total=size)
    for (
        origin_real_img,
        origin_similar_img,
        origin_wrong_img,
        spec,
        spec_len,
        raw_audio
    ) in pbar:
        # origin_real_img, origin_similar_img, origin_wrong_img, spec, spec_len
        batch = (
            origin_real_img.to(device),
            origin_similar_img.to(device),
            origin_wrong_img.to(device),
            spec.to(device),
            spec_len.to(device),
            raw_audio
        )

        D_loss = update_D(
            models,
            batch,
            optimizers["disc"],
            schedulers["disc"],
            criterions,
            specific_params,
            device,
        )
        RS_loss = update_RS(
            models,
            batch,
            optimizers["rs"],
            schedulers["rs"],
            criterions,
            specific_params,
            device,
        )
        G_loss, KL_loss = update_G(
            models,
            batch,
            optimizers["gen"],
            schedulers["gen"],
            criterions,
            specific_params,
            device,
        )

        if log_wandb:
            wandb.log({"train/G_loss": G_loss})
            wandb.log({"train/D_loss": D_loss})
            wandb.log({"train/KL_loss": KL_loss})
            wandb.log({"train/RS_loss": RS_loss})
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/lr-OneCycleLR_G": schedulers["gen"].get_last_lr()[0]})

        pbar.set_description(
            f"[Epoch: {epoch}] G_Loss: {G_loss:.2f} | D_Loss: {D_loss:.2f} | RS_loss: {RS_loss:.2f}"
        )
