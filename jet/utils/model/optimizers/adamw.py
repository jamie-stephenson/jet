from torch.optim import AdamW

def get_optimizer(model, weight_decay, cuda):
    optimizer = AdamW(params=model.parameters(),weight_decay=weight_decay, fused=cuda) 
    return optimizer