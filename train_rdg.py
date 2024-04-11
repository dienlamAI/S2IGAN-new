import argparse

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

import wandb
from data.data_collator import rdg_collate_fn
from data.dataset import RDGDataset
from s2igan.loss import KLDivergenceLoss, RSLoss
from s2igan.rdg import (
    DenselyStackedGenerator,
    DiscriminatorFor64By64,
    DiscriminatorFor128By128,
    DiscriminatorFor256By256,
    RelationClassifier,
)
from s2igan.rdg.utils import rdg_train_epoch
from s2igan.sen import ImageEncoder, SpeechEncoder
from s2igan.utils import set_non_grad

config_path = "conf"
config_name = "rdg_config"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    if cfg.experiment.log_wandb:
        wandb.init(project="speech2image", name="RDG")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_gpu = torch.cuda.device_count() > 1
    device_ids = list(range(torch.cuda.device_count()))

    train_set = RDGDataset(**cfg.data.train)
    test_set = RDGDataset(**cfg.data.test)

    bs = cfg.data.general.batch_size
    nwkers = cfg.data.general.num_workers
    train_dataloader = DataLoader(
        train_set, bs, shuffle=True, num_workers=nwkers, collate_fn=rdg_collate_fn
    )
    test_dataloder = DataLoader(
        test_set, bs, shuffle=False, num_workers=nwkers, collate_fn=rdg_collate_fn
    )

    generator = DenselyStackedGenerator(**cfg.model.generator)
    discrminator_64 = DiscriminatorFor64By64(**cfg.model.discriminator)
    discrminator_128 = DiscriminatorFor128By128(**cfg.model.discriminator)
    discrminator_256 = DiscriminatorFor256By256(**cfg.model.discriminator)
    relation_classifier = RelationClassifier(**cfg.model.relation_classifier)
    image_encoder = ImageEncoder(**cfg.model.image_encoder)
    speech_encoder = SpeechEncoder(**cfg.model.speech_encoder)

    if cfg.ckpt.image_encoder:
        print("Loading Image Encoder state dict...")
        print(image_encoder.load_state_dict(torch.load(cfg.ckpt.image_encoder)))
    if cfg.ckpt.speech_encoder:
        print("Loading Speech Encoder state dict...")
        print(speech_encoder.load_state_dict(torch.load(cfg.ckpt.speech_encoder)))
        set_non_grad(image_encoder)
        set_non_grad(speech_encoder)

    if multi_gpu:
        generator = nn.DataParallel(generator, device_ids=device_ids)
        discrminator_64 = nn.DataParallel(discrminator_64, device_ids=device_ids)
        discrminator_128 = nn.DataParallel(discrminator_128, device_ids=device_ids)
        discrminator_256 = nn.DataParallel(discrminator_256, device_ids=device_ids)
        relation_classifier = nn.DataParallel(
            relation_classifier, device_ids=device_ids
        )
        image_encoder = nn.DataParallel(image_encoder, device_ids=device_ids)
        speech_encoder = nn.DataParallel(speech_encoder, device_ids=device_ids)

    generator = generator.to(device)
    discrminator_64 = discrminator_64.to(device)
    discrminator_128 = discrminator_128.to(device)
    discrminator_256 = discrminator_256.to(device)
    relation_classifier = relation_classifier.to(device)
    image_encoder = image_encoder.to(device)
    speech_encoder = speech_encoder.to(device)

    # try:
    #     image_encoder = torch.compile(image_encoder)
    #     discrminator_64 = torch.compile(discrminator_64)
    #     discrminator_128 = torch.compile(discrminator_128)
    #     discrminator_256 = torch.compile(discrminator_256)
    #     relation_classifier = torch.compile(relation_classifier)
    #     image_encoder = torch.compile(image_encoder)
    #     speech_encoder = torch.compile(speech_encoder)
    # except:
    #     print("Can't activate Pytorch 2.0")

    discriminators = {
        64: discrminator_64,
        128: discrminator_128,
        256: discrminator_256,
    }

    models = {
        "gen": generator,
        "disc": discriminators,
        "rs": relation_classifier,
        "ied": image_encoder,
        "sed": speech_encoder,
    }

    optimizer_generator = torch.optim.AdamW(generator.parameters(), **cfg.optimizer)
    optimizer_discrminator = {
        key: torch.optim.AdamW(discriminators[key].parameters(), **cfg.optimizer)
        for key in discriminators.keys()
    }
    optimizer_rs = torch.optim.AdamW(relation_classifier.parameters(), **cfg.optimizer)
    optimizers = {
        "gen": optimizer_generator,
        "disc": optimizer_discrminator,
        "rs": optimizer_rs,
    }

    steps_per_epoch = len(train_dataloader)
    sched_dict = dict(
        epochs=cfg.experiment.max_epoch,
        steps_per_epoch=steps_per_epoch,
        max_lr=cfg.optimizer.lr,
        pct_start=cfg.scheduler.pct_start,
    )
    schedulers = {
        "gen": torch.optim.lr_scheduler.OneCycleLR(optimizer_generator, **sched_dict),
        "disc": {
            key: torch.optim.lr_scheduler.OneCycleLR(
                optimizer_discrminator[key], **sched_dict
            )
            for key in discriminators.keys()
        },
        "rs": torch.optim.lr_scheduler.OneCycleLR(optimizer_rs, **sched_dict),
    }

    criterions = {
        "kl": KLDivergenceLoss().to(device),
        "rs": RSLoss().to(device),
        "bce": nn.BCELoss().to(device),
        "ce": nn.CrossEntropyLoss().to(device),
    }

    log_wandb = cfg.experiment.log_wandb
    specific_params = cfg.experiment.specific_params
    if cfg.experiment.train:
        for epoch in range(cfg.experiment.max_epoch):
            train_result = rdg_train_epoch(
                models,
                train_dataloader,
                optimizers,
                schedulers,
                criterions,
                specific_params,
                device,
                epoch,
                log_wandb,
            )
    print("Train result:", train_result)


if __name__ == "__main__":
    load_dotenv()
    main()
