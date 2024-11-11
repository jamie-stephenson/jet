from torch.optim import Adam

def get_optimizer(model, weight_decay):
    optimizer = Adam(params=model.parameters(),weight_decay=weight_decay) 
    return optimizer