from torch.optim.lr_scheduler import OneCycleLR

def get_lr_scheduler(optimizer, args):
    scheduler = OneCycleLR(optimizer, args.lr_max, args.max_iter//args.grad_accumulation_steps) 
    return scheduler