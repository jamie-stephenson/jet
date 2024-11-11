from torch.optim import SGD

def get_optimizer(model, weight_decay):
    optimizer = SGD(params=model.parameters(),weight_decay=weight_decay) 
    return optimizer