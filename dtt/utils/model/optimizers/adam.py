from torch.optim import Adam

def get_optimizer(
    model, 
    weight_decay,
    fused,
    betas = (0.9,0.999) 
):
    optimizer = Adam(
        params=model.parameters(),
        betas=betas,
        weight_decay=weight_decay,
        fused=fused
    ) 

    return optimizer