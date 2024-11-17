from torch.optim.lr_scheduler import ExponentialLR

def get_lr_scheduler(optimizer, gamma):
    scheduler = ExponentialLR(optimizer, gamma) 
    return scheduler