from utils.dist import FnJoinable

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.algorithms.join import Join

import time
import wandb

def train(model: DDP,tokenizer,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args):

    model.train()

    val = FnJoinable(
        evaluate,
        model.device, 
        model.process_group,
        model=model,
        loader=eval_dataloader,
        args=args
    )

    dist_cm = Join([model,val]) 

    batch = 0
    eff_batch = 0

    start_time = time.time()

    with dist_cm:
        for epoch in range(args.epochs):

            if args.rank == 0:
                if args.no_wandb:
                    print("-"*40)
                    print(f"Epoch {epoch}")
                else:
                    wandb.log({"Epoch": epoch})
                                
            for x, y in train_dataloader:

                batch += 1
            
                x, y = x.to(args.device), y.to(args.device)

                if args.world_size > 1:
                    model.require_backward_grad_sync = (batch%args.grad_accumulation_steps == 0) # If true, `loss.backward()` will trigger gradient sync

                logits = model(x)

                loss = F.cross_entropy(logits.view(-1,args.vocab_size), y.view(-1))
                
                loss.backward()

                if batch%args.grad_accumulation_steps == 0: # Only take step when gradients have accumulated

                    # -----------VALIDATION-&-LOGGING--------------   
                
                    if eff_batch % args.eff_batch_per_log == 0 and args.rank == 0:
                        current_time = int(time.time()) - int(start_time) 
                        if args.no_wandb:
                            print("-"*40)
                            print(f"Effective Batch {eff_batch:.0f}")
                            print("-"*40)
                            print(f"Training has been executing for {current_time} seconds.")
                            print(f"Current training loss is: {loss:.2f}")
                        else:
                            wandb.log({
                                "Train Loss": loss.item(),
                                "Learning Rate": optimizer.param_groups[0]['lr'],
                                "Time": current_time,
                                "Effective Batch Number": eff_batch
                            })
                        

                    if args.log_per_val != -1 and eff_batch % (args.log_per_val*args.eff_batch_per_log) == 0:  
                        val_loss = val()
                    
                        if args.rank == 0:
                        
                            if args.no_wandb:
                                print(f"Current validation loss is: {val_loss:.2f}")
                            else:
                                wandb.log({
                                    "Val Loss": val_loss,
                                })
                                print("-"*40)
                                print(f"Effective Batch {eff_batch:.0f}")
                                print("-"*40)
                            
                            print('Sample Output:')
                            response = generate(model,tokenizer,args.val_prompt,args.temp,args.device)
                            print(args.val_prompt+response)
                            
                    # ----------------------------------------------       

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    eff_batch += 1     

    train_time = int(time.time()) - int(start_time)

    return model

def evaluate(model, loader, args):
    model_mode = model.training
    model.eval()

    loss_sum = torch.tensor(0,dtype=torch.float32,device=args.device)
    nsamples = torch.tensor(0,dtype=torch.float32,device=args.device)

    with torch.no_grad():
        for x, y in loader:

            x, y = x.to(args.device), y.to(args.device)

            logits = model.module.forward(x) # No direct call to avoid join hooks

            loss_sum += F.cross_entropy(logits.view(-1,args.vocab_size), y.view(-1))
            nsamples += 1

        dist.all_reduce(loss_sum)
        dist.all_reduce(nsamples)

    loss = loss_sum/nsamples

    if model_mode:
        model.train()

    return loss

def generate(model,tokenizer,string,temp,device='cpu'):

    model_mode = model.training
    model.eval()

    # Hacky way to infer seq_len without needing args, allows use of `generate` outside training scenario
    seq_len = next(v.num_embeddings for k,v in model.module.named_modules() if 'positional' in k)

    output = []

    with torch.no_grad():
        x = torch.tensor(tokenizer.encode(string)[-seq_len:]).view(1,-1).to(device) # (1,seq_len)
        for _ in range(20): #TODO Implement special tokens so this doesn't need an arbitrary limit.    
            prob = F.softmax(model.module.forward(x)[0, -1] / temp,dim=0) # (vocab_size,)
            topk_prob, topk_tokens = torch.topk(prob, 50, dim=-1)
            next_idx = torch.multinomial(topk_prob, 1).view(1,-1) # (1,1)
            next_token = topk_tokens[next_idx]
            x = torch.cat((x, next_token),dim=1)[:,-seq_len:] # (1,seq_len)
            output.append(next_token.item())

    if model_mode:
        model.train()

    return tokenizer.decode(output)