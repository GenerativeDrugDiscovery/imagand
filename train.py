import argparse
import os
import torch
from dataloader import PKDataloader

from imagand import SDT, EMA

from torch import nn
import math

from diffusers.optimization import get_scheduler
from tqdm import tqdm

from utils import *

from sklearn.metrics import mean_squared_error
import csv

from diffusion import DDIMScheduler

parser = argparse.ArgumentParser()

parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--wd', dest='wd', type=float, default=5e-2)
parser.add_argument('--warmup', dest='warmup', type=int, default=200)
parser.add_argument('--n_timesteps', dest='n_timesteps', type=int, default=2000)
parser.add_argument('--n_inference_timesteps', dest='n_inference_timesteps', type=int, default=150)
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=3000)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
parser.add_argument('--gamma', dest='gamma', type=float, default=0.994)

parser.add_argument('--data_dir', dest='data_dir', type=str, default='./data')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='./output')

parser.add_argument('--noise_type', choices=['gaussian', 'uniform', 'power', 'none'], default='none')
parser.add_argument('--embed_model', choices=[
    't5',
    'deberta',
    'chemberta_zinc',
    'chemberta_10m'
], default='t5')

args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

dataloader = PKDataloader(
    args.embed_model,
    args.data_dir
)

trainset = dataloader.dataset
dmss = trainset.dmss
trainset, valset = torch.utils.data.random_split(trainset, [0.9,0.10])
# valset, testset = torch.utils.data.random_split(valset, [0.5,0.5])

print(trainset[1]['gt'].shape)
print(trainset[1]['ma'].shape)
print(trainset[1]['ft'].shape)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)  

steps_per_epoch = len(trainset)

device = "cuda"

model = SDT(
    time_dim = 64,
    cond_size = 768,
    patch_size = 16,
    y_dim = 12,
    dim = 256,
    depth = 12,
    heads = 16,
    mlp_dim = 768,
    dropout = 0.1,
    emb_dropout = 0.1
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

def train(model, ema, gamma, dataloader, noise_scheduler, optimizer, lr_scheduler):
    model.train()
    running_loss = 0
    global_step = 0
    mse_loss = nn.MSELoss(reduction='none')
    for i, batch in enumerate(tqdm(dataloader)):
        ft = batch['ft'].to(device).float()
        gt = batch['gt'].to(device).float()
        mask = batch['ma'].to(device)
        bs = ft.shape[0]

        noise = sample_noise(bs, dmss)
        noise = torch.tensor(noise).to(device).float()
        timesteps = torch.randint(0,
                                noise_scheduler.num_train_timesteps,
                                (bs,),
                                device=device).long()

        noisy_gt = noise_scheduler.add_noise(gt, noise, timesteps)

        optimizer.zero_grad()
        noise_pred = model(ft, noisy_gt, timesteps)

        loss = mse_loss(noise_pred, noise)
        loss = (loss * mask.float()).sum()
        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements
        mse_loss_val.backward()
        optimizer.step()
        lr_scheduler.step()

        ema.update_params(gamma)
        gamma = ema.update_gamma(global_step)

        running_loss += mse_loss_val.item()
        global_step += 1
    return running_loss/global_step

def evaluate(e, ema, dataloader, noise_scheduler, n_inference_timesteps):
    ema.ema_model.eval()
    before_mse = 0
    running_mse = 0
    global_step = 0
    vals = {}
    device = 'cuda'
    ema.ema_model.to(device)
    noise_scheduler.set_timesteps(n_inference_timesteps)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            sm = batch['sm']
            mask = batch['ma']
            ft = batch['ft'].to(device).float()
            gt = batch['gt'].to(device).float()
            bs = ft.shape[0]

            ys = sample_noise(bs, dmss)
            ys = torch.tensor(ys).to(device).float()
            timestep = torch.tensor([n_inference_timesteps], device=device).long()
            #ys[mask] = noise_scheduler.add_noise(gt[mask], ys[mask], timestep)

            raw_mse = mean_squared_error(gt[mask].flatten().cpu(), ys[mask].flatten().cpu())
            # non_zero_elements = mask.sum()
            # raw_mse = raw_mse / non_zero_elements

            generated_ys = noise_scheduler.generate(
                ema.ema_model,
                ft,
                ys,
                num_inference_steps=n_inference_timesteps,
                eta=0.01,
                use_clipped_model_output=True,
                device = device
            )

            mse = mean_squared_error(gt[mask].flatten().cpu(), generated_ys[mask].flatten().cpu())
            # mse = mse / non_zero_elements

            for s, g in zip(sm, list(generated_ys.cpu().numpy())):
                vals[s] = g
            
            before_mse += raw_mse
            running_mse += mse
            global_step += 1

    with open(args.save_dir+'{}_dict.csv'.format(e), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in vals.items():
            writer.writerow([key, value])

    return running_mse / global_step, before_mse / global_step

total_num_steps = (steps_per_epoch * args.num_epochs)

ema = EMA(model, args.gamma, total_num_steps)
ns = DDIMScheduler(num_train_timesteps=args.n_timesteps,
                                beta_start=0.,
                                beta_end=0.7,
                                beta_schedule="cosine")

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=total_num_steps,
    )

l = ""
best_mse = 1
for e in range(args.num_epochs):
    loss = train(model, ema, args.gamma, trainloader, ns, optimizer, lr_scheduler)
    if (e % 10 == 0) and (e > 0):
        mse, bmse = evaluate(e, ema, valloader, ns, args.n_inference_timesteps)

        print(e, "avgloss {}, avgvalmse {}, beforemse: {}".format(loss, mse, bmse))
        l += str({
            "type": "val",
            "e":e,
            "avgloss":loss,
            "avgvalmse":mse,
            "beforemse": bmse
        }) + "\n"

        if mse < best_mse:
            best_mse = mse
            torch.save({
                'e': e,
                'ema_model': ema.ema_model.state_dict(),
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.save_dir+"best_model.pt")
    else:
        print(e, "avgloss {}".format(loss))
        l += str({
            "type": "train",
            "e":e,
            "avgloss":loss,
        }) + "\n"

    with open(args.save_dir+'output.txt', 'w') as file:
        file.write(l)