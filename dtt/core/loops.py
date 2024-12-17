from dtt.utils import Config
from dtt.utils.dist import FnJoinable

from bpekit import Tokenizer
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.distributed.algorithms.join import Join
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb

from time import time


def train(
    model: DistributedDataParallel,
    tokenizer: Tokenizer,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    cfg: Config
) -> DistributedDataParallel:

    model.train()

    val = FnJoinable(
        evaluate,
        model.device, 
        model.process_group,
        model=model,
        loader=eval_dataloader,
        cfg=cfg
    )

    dist_cm = Join([model,val]) 

    batch = 0
    eff_batch = 0
    tokens_per_log = cfg.batch_size*cfg.grad_accumulation_steps*cfg.n_ctx*cfg.world_size*cfg.eff_batch_per_log
    t0 = t_log = time()

    with dist_cm:
        for epoch in range(cfg.epochs):

            if cfg.rank == 0:
                if cfg.wandb:
                    wandb.log({"Epoch": epoch})
                else:
                    print("-"*40)
                    print(f"Epoch {epoch}")
                    print("-"*40)
                                
            for x, y in train_dataloader:

                batch += 1
            
                x, y = x.to(cfg.device), y.to(cfg.device)

                if cfg.world_size > 1:
                    model.require_backward_grad_sync = (batch%cfg.grad_accumulation_steps == 0) # If true, `loss.backward()` will trigger gradient sync

                with torch.autocast(cfg.device_type,torch.bfloat16,enabled=cfg.autocast):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1,cfg.vocab_size), y.view(-1))
                
                loss.backward()

                if batch%cfg.grad_accumulation_steps == 0: # Only take step when gradients have accumulated

                    # -----------VALIDATION-&-LOGGING--------------   
                
                    if eff_batch % cfg.eff_batch_per_log == 0 and cfg.rank == 0: 
                        t1 = time()
                        dt = t1-t_log
                        t_log = t1
                                            
                        if cfg.wandb:
                            wandb.log({
                                "Train Loss": loss.item(),
                                "Learning Rate": optimizer.param_groups[0]['lr'],
                                "Time": t1 - t0,
                                "Effective Batch Number": eff_batch,
                                "Tokens per Second": tokens_per_log/dt
                            })
                        else:
                            print(f"eff batch {eff_batch:.0f} | loss: {loss.item():.2f} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_log/dt:.2f}")
                        

                    if cfg.log_per_val and eff_batch % (cfg.log_per_val*cfg.eff_batch_per_log) == 0:  
                        val_loss = val()
                    
                        if cfg.rank == 0:
                        
                            if cfg.wandb:
                                wandb.log({
                                    "Val Loss": val_loss,
                                })
                                print("-"*40)
                                print(f"Effective Batch {eff_batch:.0f}")
                                print("-"*40)
                            else:
                                print(f"Current validation loss is: {val_loss:.2f}")
                            
                            print('Sample Output:')
                            response = generate(model,tokenizer,cfg.val_prompt,cfg.temp,cfg.device)
                            print(cfg.val_prompt+response)
                            
                    # ----------------------------------------------       

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    eff_batch += 1     

    train_time = int(time()) - int(t0)

    return model

def evaluate(
    model: DistributedDataParallel, 
    loader: DataLoader,
    cfg: Config
):
    model_mode = model.training
    model.eval()


    with torch.no_grad():
        loss_sum = torch.tensor(0,dtype=torch.float32,device=cfg.device)
        nsamples = torch.tensor(0,dtype=torch.float32,device=cfg.device)
        for x, y in loader:

            x, y = x.to(cfg.device), y.to(cfg.device)

            with torch.autocast(cfg.device_type,torch.bfloat16,enabled=cfg.autocast):
                logits = model.module.forward(x) # No direct call to avoid join hooks
                loss_sum += F.cross_entropy(logits.view(-1,cfg.vocab_size), y.view(-1))
            
            nsamples += 1

        dist.all_reduce(loss_sum)
        dist.all_reduce(nsamples)

    loss = loss_sum/nsamples

    if model_mode:
        model.train()

    return loss

def generate(
    model: DistributedDataParallel,
    tokenizer: Tokenizer,
    string: str,
    temp: float,
    device: str = 'cpu'
) -> str:

    model_mode = model.training
    model.eval()

    # Hacky way to infer seq_len without needing cfg, allows use of `generate` outside training scenario
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