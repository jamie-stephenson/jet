import torch.nn.functional as F
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
import time
from datetime import datetime
import os
from tqdm import tqdm
import wandb

def train(model,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args):
    model.train()

    start_time = time.time()

    prof_dir = f"./profile/train/{args.encoded_format}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(prof_dir,exist_ok=True)

    sched = schedule(
        wait=4,
        warmup=2,
        active=8,
        repeat=4
    )

    prof = profile(
        activities=[ProfilerActivity.CPU],
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(prof_dir)
    )

    with prof as p:

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
            p.step()

            if i % (args.eff_batch_per_log*args.grad_accumulation_steps) == 0 and args.rank == 0 and i > 0:
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
   
    train_time = int(time.time()) - int(start_time)

    return model
