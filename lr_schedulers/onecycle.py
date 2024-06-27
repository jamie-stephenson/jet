from torch.optim.lr_scheduler import OneCycleLR

def get_lr_scheduler(optimizer, steps_per_epoch, args):
    scheduler = OneCycleLR(optimizer, args.lr_max, epochs=args.epochs, steps_per_epoch=steps_per_epoch) 
    return scheduler