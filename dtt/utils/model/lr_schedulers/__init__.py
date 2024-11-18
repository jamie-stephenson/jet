import importlib

def get_lr_scheduler(name, optimizer, *args, **kwargs):
    
    lr_scheduler_module = importlib.import_module(f".{name}", package="dtt.utils.model.lr_schedulers")
    lr_scheduler = lr_scheduler_module.get_lr_scheduler(optimizer, *args, **kwargs)

    return lr_scheduler