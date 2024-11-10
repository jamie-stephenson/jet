import importlib

def get_lr_scheduler(name, optimizer, steps_per_epoch, args):
    
    lr_scheduler_module = importlib.import_module(f".{name}", package="utils.model.lr_schedulers")
    lr_scheduler = lr_scheduler_module.get_lr_scheduler(optimizer, steps_per_epoch, args)

    return lr_scheduler