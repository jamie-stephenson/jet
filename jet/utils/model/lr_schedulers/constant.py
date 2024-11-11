from torch.optim.lr_scheduler import LambdaLR

def get_lr_scheduler(optimizer, lr_max):
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x: lr_max)
    return scheduler