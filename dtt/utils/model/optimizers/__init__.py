import importlib

def get_optimizer(name, model, *args, **kwargs):
    
    optimizer_module = importlib.import_module(f".{name}", package="dtt.utils.model.optimizers")
    optimizer = optimizer_module.get_optimizer(model, *args, **kwargs)

    return optimizer