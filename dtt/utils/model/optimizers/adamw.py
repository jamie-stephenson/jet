from torch.optim import AdamW

def get_optimizer(
    model,
    weight_decay, 
    fused,
    betas = (0.9,0.999) 
):
    
    optimizer = AdamW(
        params=model.parameters(),
        betas=betas,
        weight_decay=weight_decay,
        fused=fused
    )

    return optimizer