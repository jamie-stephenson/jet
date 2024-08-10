from torch.optim import AdamW

def get_optimizer(model, args):
    optimizer = AdamW(params=model.parameters(),weight_decay=args.weight_decay) 
    return optimizer