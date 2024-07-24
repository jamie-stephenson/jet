from torch.optim.lr_scheduler import LambdaLR

def get_lr_scheduler(optimizer, steps_per_epoch, args):
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x: args.lr_max)
    return scheduler