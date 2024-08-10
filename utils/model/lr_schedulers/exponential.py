from torch.optim.lr_scheduler import ExponentialLR

def get_lr_scheduler(optimizer, steps_per_epoch, args):
    scheduler = ExponentialLR(optimizer, args.lr_gamma) 
    return scheduler