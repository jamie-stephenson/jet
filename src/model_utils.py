import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
from tqdm import tqdm
import wandb
import os

def train(model,tokenizer,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args):

    model.train()

    start_time = time.time()

    for epoch in range(args.epochs):
    
        if args.rank == 0:
            if args.no_wandb:
                print("-"*40)
                print(f"Epoch {epoch}")
            else:
                wandb.log({"Epoch": epoch})

        for i,(x, y) in enumerate(train_dataloader):
        
            x, y = x.to(args.device), y.to(args.device)

            if args.world_size > 1:
                model.require_backward_grad_sync = ((i+1)%args.grad_accumulation_steps == 0) # If true, `loss.backward()` will trigger gradient sync

            logits = model(x)

            loss = F.cross_entropy(logits.view(-1,args.vocab_size), y.view(-1))
            
            loss.backward()

            if (i+1)%args.grad_accumulation_steps == 0: # Only take step when gradients have accumulated
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # -----------VALIDATION-&-LOGGING--------------   
        
            if i % (args.eff_batch_per_log*args.grad_accumulation_steps) == 0 and args.rank == 0:
                current_time = int(time.time()) - int(start_time)
                eff_batch = i/args.grad_accumulation_steps
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
                

            if i % (args.log_per_val*args.eff_batch_per_log*args.grad_accumulation_steps) == 0:  
                val_loss = evaluate(model, eval_dataloader, args)
            
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

    train_time = int(time.time()) - int(start_time)

    return model

def evaluate(model, dataloader, args):

    model_mode = model.training
    model.eval()

    loss_sum = 0

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(args.device), y.to(args.device)

            logits = model(x)

            loss_sum += F.cross_entropy(logits.view(-1,args.vocab_size), y.view(-1))

        loss = loss_sum/len(dataloader)
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    if model_mode:
        model.train()

    return loss

def generate(model,tokenizer,string,temp,device='cpu'):

    model_mode = model.training
    model.eval()

    # Hacky way to infer seq_len without needing args, allows use of `generate` outside training scenario
    seq_len = list(model.modules())[int(os.getenv('WORLD_SIZE',1))>1][0].positional_embedding.num_embeddings

    output = []

    with torch.no_grad():
        x = torch.tensor(tokenizer.encode(string)).to(device)
        for _ in range(20): #TODO Implement special tokens so this doesn't need an arbitrary limit.    
            prob = F.softmax(model(x)[0, -1, :] / temp)
            next_token = torch.multinomial(prob, 1)
            x = torch.cat(x, next_token)[-seq_len:]
            output.append(next_token.item())

    if model_mode:
        model.train()

    return tokenizer.decode(output)