import wandb


def set_non_grad(model):
    for param in model.parameters():
        param.requires_grad = False
