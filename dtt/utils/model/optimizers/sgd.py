from torch.optim import SGD

def get_optimizer(
    model, 
    momentum, 
    weight_decay,
    fused
):
    
    optimizer = SGD(
        params=model.parameters(),
        momentum=momentum,
        weight_decay=weight_decay,
        fused=fused
    ) 
    
    return optimizer