from torch.optim import Adam

def get_optimizer(model, args):
    optimizer = Adam(params=model.parameters(),weight_decay=args.weight_decay) 
    return optimizer