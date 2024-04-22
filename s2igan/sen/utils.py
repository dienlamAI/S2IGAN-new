import torch
from tqdm import tqdm

import wandb


def sen_train_epoch(
    image_encoder,
    speech_encoder,
    classifier,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    log_wandb: bool = False,
):
    size = len(dataloader)
    run_loss = 0
    run_img_acc = 0
    run_speech_acc = 0
    pbar = tqdm(dataloader, total=size)
    for (imgs, specs, len_specs, labels) in pbar:
        imgs, specs, len_specs, labels = (
            imgs.to(device),
            specs.to(device),
            len_specs.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        V = image_encoder(imgs)
        A = speech_encoder(specs, len_specs)
        cls_img = classifier(V)
        cls_speech = classifier(A)

        match_loss, dist_loss, loss = criterion(V, A, cls_img, cls_speech, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        loss = loss.item()

        img_acc = (cls_img.argmax(-1) == labels).sum() / labels.size(0) * 100
        speech_acc = (cls_speech.argmax(-1) == labels).sum() / labels.size(0) * 100

        img_acc = img_acc.item()
        speech_acc = speech_acc.item()

        run_loss += loss
        run_img_acc += img_acc
        run_speech_acc += speech_acc

        if log_wandb:
            wandb.log({"train/sen_loss": loss})
            wandb.log({"train/matching_loss": match_loss.item()})
            wandb.log({"train/distinctive_loss": dist_loss.item()})
            wandb.log({"train/image_accuracy": img_acc})
            wandb.log({"train/speech_accuracy": speech_acc})
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/lr-OneCycleLR": scheduler.get_last_lr()[0]})

        pbar.set_description(
            f"[Epoch: {epoch}] Loss: {loss:.2f} | Image Acc: {img_acc:.2f}% | Speech Acc: {speech_acc:.2f}%"
        )

    return {
        "loss": run_loss / size,
        "img_acc": run_img_acc / size,
        "speech_acc": run_speech_acc / size,
    }


def sen_eval_epoch(
    image_encoder,
    speech_encoder,
    classifier,
    dataloader,
    criterion,
    device,
    epoch,
    log_wandb: bool = False,
):
    size = len(dataloader)
    run_loss = 0
    run_img_acc = 0
    run_speech_acc = 0
    pbar = tqdm(dataloader, total=size)
    with torch.no_grad():
        for (imgs, specs, len_specs, labels) in pbar:
            imgs, specs, len_specs, labels = (
                imgs.to(device),
                specs.to(device),
                len_specs.to(device),
                labels.to(device),
            )

            V = image_encoder(imgs)
            A = speech_encoder(specs, len_specs)
            cls_img = classifier(V)
            cls_speech = classifier(A)

            match_loss, dist_loss, loss = criterion(V, A, cls_img, cls_speech, labels)

            loss = loss.item()

            img_acc = (cls_img.argmax(-1) == labels).sum() / labels.size(0) * 100
            speech_acc = (cls_speech.argmax(-1) == labels).sum() / labels.size(0) * 100

            img_acc = img_acc.item()
            speech_acc = speech_acc.item()

            run_loss += loss
            run_img_acc += img_acc
            run_speech_acc += speech_acc

            if log_wandb:
                wandb.log({"val/sen_loss": loss})
                wandb.log({"val/matching_loss": match_loss.item()})
                wandb.log({"val/distinctive_loss": dist_loss.item()})
                wandb.log({"val/image_accuracy": img_acc})
                wandb.log({"val/speech_accuracy": speech_acc})

            pbar.set_description(
                f"[Epoch: {epoch}] Loss: {loss:.2f} | Image Acc: {img_acc:.2f}% | Speech Acc: {speech_acc:.2f}%"
            )

    return {
        "loss": run_loss / size,
        "img_acc": run_img_acc / size,
        "speech_acc": run_speech_acc / size,
    }

# sed

def sed_train_epoch( 
    speech_encoder,
    classifier,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    log_wandb: bool = False,
):
    size = len(dataloader)
    run_loss = 0
    run_speech_acc = 0
    pbar = tqdm(dataloader, total=size)
    for (specs, len_specs, labels) in pbar:
        specs, len_specs, labels = (
            specs.to(device),
            len_specs.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        A = speech_encoder(specs, len_specs)
        cls_speech = classifier(A)

        # dist_loss, loss = criterion(cls_speech, labels)
        dist_loss, loss = criterion(cls_speech.unsqueeze(0), labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        loss = loss.item()

        speech_acc = (cls_speech.argmax(-1) == labels).sum() / labels.size(0) * 100

        speech_acc = speech_acc.item()

        run_loss += loss
        run_speech_acc += speech_acc

        if log_wandb:
            wandb.log({"train/sen_loss": loss})
            wandb.log({"train/distinctive_loss": dist_loss.item()})
            wandb.log({"train/speech_accuracy": speech_acc})
            wandb.log({"train/epoch": epoch})
            wandb.log({"train/lr-OneCycleLR": scheduler.get_last_lr()[0]})

        pbar.set_description(
            f"[Epoch: {epoch}] Loss: {loss:.2f} | Speech Acc: {speech_acc:.2f}%"
        )

    return {
        "loss": run_loss / size,
        "speech_acc": run_speech_acc / size,
    }


def sed_eval_epoch( 
    speech_encoder,
    classifier,
    dataloader,
    criterion,
    device,
    epoch,
    log_wandb: bool = False,
):
    size = len(dataloader)
    run_loss = 0
    run_speech_acc = 0
    pbar = tqdm(dataloader, total=size)
    with torch.no_grad():
        for (specs, len_specs, labels) in pbar:
            specs, len_specs, labels = (
                specs.to(device),
                len_specs.to(device),
                labels.to(device),
            )

            A = speech_encoder(specs, len_specs)
            cls_speech = classifier(A)

            # dist_loss, loss = criterion( cls_speech, labels)
            dist_loss, loss = criterion(cls_speech.unsqueeze(0), labels)

            loss = loss.item()

            speech_acc = (cls_speech.argmax(-1) == labels).sum() / labels.size(0) * 100

            speech_acc = speech_acc.item()

            run_loss += loss
            run_speech_acc += speech_acc

            if log_wandb:
                wandb.log({"val/sen_loss": loss})
                wandb.log({"val/distinctive_loss": dist_loss.item()})
                wandb.log({"val/speech_accuracy": speech_acc})

            pbar.set_description(
                f"[Epoch: {epoch}] Loss: {loss:.2f} | Speech Acc: {speech_acc:.2f}%"
            )

    return {
        "loss": run_loss / size,
        "speech_acc": run_speech_acc / size,
    }
