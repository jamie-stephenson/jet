import importlib

def get_optimizer(name, model, args):
    
    optimizer_module = importlib.import_module("optimizers." + name, package=".")
    optimizer = optimizer_module.get_optimizer(model, args)

    return optimizer