import argparse

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
import os
import wandb
from data.data_collator import sed_collate_fn
from data.dataset import SEDDataset
from s2igan.loss import SEDLoss
from s2igan.sen.sed import  SpeechEncoder, SpeechEncoderNew
from s2igan.sen.utils import sed_train_epoch, sed_eval_epoch

config_path = "conf"
config_name = "sen_config"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    bs = cfg.data.general.batch_size
    attn_heads = cfg.model.speech_encoder.attn_heads
    attn_dropout = cfg.model.speech_encoder.attn_dropout
    rnn_dropout = cfg.model.speech_encoder.rnn_dropout
    lr = cfg.optimizer.lr
    wandb.init(project="speech2image", name=f"SEN_bs{bs}_lr{lr}_attn{attn_heads}_ad{attn_dropout}_rd{rnn_dropout}_{cfg.kaggle.user}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_gpu = torch.cuda.device_count() > 1
    device_ids = list(range(torch.cuda.device_count()))

    train_set = SEDDataset(**cfg.data.train)
    test_set = SEDDataset(**cfg.data.test)
 
    nwkers = cfg.data.general.num_workers
    train_dataloader = DataLoader(
        train_set, bs, shuffle=True, num_workers=nwkers, collate_fn=sed_collate_fn
    )
    test_dataloder = DataLoader(
        test_set, bs, shuffle=False, num_workers=nwkers, collate_fn=sed_collate_fn
    )
 
    # speech_encoder = SpeechEncoder(**cfg.model.speech_encoder)
    speech_encoder = SpeechEncoderNew()
    classifier = nn.Linear(**cfg.model.classifier)
    nn.init.xavier_uniform_(classifier.weight.data)
    if cfg.ckpt.speech_encoder:
        print("Loading Speech Encoder state dict...")
        print(speech_encoder.load_state_dict(torch.load(cfg.ckpt.speech_encoder)))

    if cfg.ckpt.classifier:
        print("Loading Classifier state dict...")
        print(classifier.load_state_dict(torch.load(cfg.ckpt.classifier)))

    if multi_gpu: 
        speech_encoder = nn.DataParallel(speech_encoder, device_ids=device_ids)
        classifier = nn.DataParallel(classifier, device_ids=device_ids)
 
    speech_encoder = speech_encoder.to(device)
    classifier = classifier.to(device)

    # try:
    #     image_encoder = torch.compile(image_encoder)
    #     speech_encoder = torch.compile(speech_encoder)
    #     classifier = torch.compile(classifier)
    # except:
    #     print("Can't activate Pytorch 2.0")

    if multi_gpu:
        model_params = ( 
            speech_encoder.module.get_params()
            + list(classifier.module.parameters())
        )
    else:
        model_params = ( 
            speech_encoder.get_params()
            + list(classifier.parameters())
        )

    optimizer = torch.optim.AdamW(model_params, **cfg.optimizer)
    scheduler = None
    if cfg.scheduler.use:
        steps_per_epoch = len(train_dataloader)
        sched_dict = dict(
            epochs=cfg.experiment.max_epoch,
            steps_per_epoch=steps_per_epoch,
            max_lr=cfg.optimizer.lr,
            pct_start=cfg.scheduler.pct_start,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **sched_dict)

    criterion = SEDLoss().to(device)

    log_wandb = cfg.experiment.log_wandb

    if cfg.experiment.train:
        for epoch in range(cfg.experiment.max_epoch):
            train_result = sed_train_epoch( 
                speech_encoder,
                classifier,
                train_dataloader,
                optimizer,
                scheduler,
                criterion,
                device,
                epoch,
                log_wandb,
            )
            eval_result = sed_eval_epoch( 
                speech_encoder,
                classifier,
                test_dataloder,
                criterion,
                device,
                epoch,
                log_wandb,
            )

            save_dir = "/kaggle/working/save_ckpt"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Tiếp tục lưu trữ trọng số của mô hình
            torch.save(speech_encoder.state_dict(), os.path.join(save_dir, "speech_encoder_SED.pt")) 
            torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
 
    print("Train result:", train_result)
    print("Eval result:", eval_result)


if __name__ == "__main__":
    load_dotenv()
    main()