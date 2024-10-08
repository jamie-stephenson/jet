import importlib

def get_optimizer(name, model, args):
    
    optimizer_module = importlib.import_module(f".{name}", package="utils.model.optimizers")
    optimizer = optimizer_module.get_optimizer(model, args)

    return optimizer