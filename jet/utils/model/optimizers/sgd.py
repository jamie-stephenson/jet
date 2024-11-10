from torch.optim import SGD

def get_optimizer(model, args):
    optimizer = SGD(params=model.parameters(),weight_decay=args.weight_decay) 
    return optimizer