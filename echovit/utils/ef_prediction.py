import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from echovit.dataset.vivitecho import ViViTecho

import math
import os
import click
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn.metrics
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from echovit.model.model import ViViT
import numpy as np
from echovit.utils import get_mean_and_std, clip_collate

@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/EE543-Term-Project/Video-Vision-Transformer/data")
@click.option("--output", type=click.Path(file_okay=False), default="output")
@click.option("--hyperparameter_dir", type=click.Path(file_okay=False), default="hyperparam_outputs")
@click.option("--run_test/--skip_test", default=True)
@click.option("--hyperparameter", type=bool, default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=128)
@click.option("--num_heads", type=int, default=8)
@click.option("--num_layers", type=int, default=10)
@click.option("--projection_dim", type=int, default=1024)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)


def run(

    data_dir,
    output,
    run_test,
    hyperparameter,
    hyperparameter_dir,
    num_epochs,
    lr,
    weight_decay,
    num_workers,
    batch_size,
    device,
    seed,
    projection_dim,
    num_heads, 
    num_layers,
    input_shape  = (3, 32, 112, 112),
    patch_size   = (32, 16, 16),
):

    os.makedirs(output, exist_ok=True)  # Ensure the base output directory exists
    if hyperparameter:
        output = hyperparameter_dir
    output = os.path.join(output, f"lr_{lr}_wd_{weight_decay}_bs_{batch_size}_nh_{num_heads}_nl_{num_layers}_pd_{projection_dim}") if hyperparameter else output
    os.makedirs(output, exist_ok=True)

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

     # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Initialize model, optimizer and criterion
    model       = ViViT(input_shape, patch_size, projection_dim, num_heads, num_layers)
    model       = nn.DataParallel(model)
    model       = model.to(device)
    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",          # loss azalınca daha iyi
#     factor=0.5,          # LR = LR * 0.5
#     patience=3,          # 3 val‑epoch sabit / yükseliyorsa düşür
#     threshold=1e-4,      # minimum gelişme eşiği
#     min_lr=1e-6,         # LR’nin düşebileceği alt sınır
# )
     
    # optimizer = torch.optim.AdamW(
    # model.parameters(),
    # lr=lr,                 # örn. 3e-4 ya da mevcut lr
    # weight_decay=weight_decay,  # ViT literatüründe 0.02 önerilir
    # betas=(0.9, 0.999),
    # eps=1e-8
    # )
    # warmup_epochs   = 5
    # total_epochs    = num_epochs
    # eta_min         = 1e-6          # son lr (cosine'in tabanı)

    # # PyTorch ≥ 1.11 kullanıyorsanız:
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[
    #         # a) Lineer warm-up: lr 0.1·lr_start → lr_start
    #         torch.optim.lr_scheduler.LinearLR(
    #             optimizer,
    #             start_factor=0.1,
    #             total_iters=warmup_epochs
    #         ),
    #         # b) Cosine decay: lr_start → eta_min
    #         torch.optim.lr_scheduler.CosineAnnealingLR(
    #             optimizer,
    #             T_max=total_epochs - warmup_epochs,
    #             eta_min=eta_min
    #         )
    #     ],
    #     milestones=[warmup_epochs]   # warm-up bittiği iter/epoch
    # )
    # Compute mean and std
    mean, std = get_mean_and_std(ViViTecho(root=data_dir, split="train",oversample=False))

    # Set up datasets and dataloaders
    dataset     = {}
    # Load datasets
    dataset["train"]   = ViViTecho(root=data_dir, split="train", mean=mean, std=std,length=32,period=2,clips=1)
    dataset["val"]     = ViViTecho(root=data_dir, split="val",  mean=mean, std=std,length=32,period=2,clips=1)
    dataset["test"]    = ViViTecho(root=data_dir, split="test", mean=mean, std=std,length=32,period=2,clips=1)


    log_file_path = os.path.join(output, "log.csv")
    # Run training and testing loops
    with open(os.path.join(log_file_path), "a") as f:

        f.write("epoch,phase,loss,r2,time,y_size,batch_size\n")
        train_losses    = []
        val_losses      = []

        for epoch in range(num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds                                   = dataset[phase]
                dataloader                           = DataLoader(ds, batch_size=batch_size, shuffle=True,num_workers=num_workers,collate_fn=clip_collate)
                print("train collate ->", dataloader.collate_fn)
                loss, yhat, y, filename              = run_epoch(model, dataloader, phase, optimizer, device,train_losses,val_losses)

                f.write("{},{},{},{},{},{},{}\n".format(
                                                            epoch,
                                                            phase,
                                                            loss,
                                                            sklearn.metrics.r2_score(y, yhat),
                                                            time.time() - start_time,
                                                            y.size,
                                                            batch_size))
                f.flush()
            if phase == 'train':    
                scheduler.step()

        if run_test:
            split = "test"
            ds = dataset[split]

            dataloader                          = DataLoader(ViViTecho(root=data_dir, split=split, mean=mean, std=std, length=32, period=2),batch_size=1,
                                                 shuffle=True,num_workers=num_workers)
            loss, yhat, y, filename             = run_epoch(model, dataloader, split, optimizer, device,train_losses=[],val_losses=[])
            # Write full performance to file
            with open(os.path.join(output, "{}_predictions.csv".format(split)), "a") as g:
                g.write("filename,true_value, predictions\n")
                for (file, pred, target) in zip(filename, yhat, y):
                        g.write("{},{},{:.4f}\n".format(file,target,pred))

                g.write("{} R2:   {:.3f} \n".format(split, sklearn.metrics.r2_score(y, yhat)))
                g.write("{} MAE:  {:.2f} \n".format(split, sklearn.metrics.mean_absolute_error(y, yhat)))
                g.write("{} RMSE: {:.2f} \n".format(split, math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
                g.flush()


    np.save(os.path.join(output, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output, "val_losses.npy"), np.array(val_losses))
    print(f"Train and validation losses saved to {output}")

def run_epoch(model, dataloader, phase, optimizer,device,train_losses, val_losses):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer

    """
    model.train(phase== "train")

    yhat = []          # Prediction
    y    = []          # Ground truth

    n    = 0           # number of videos processed
    s1   = 0           # sum of ground truth EF
    s2   = 0           # Sum of ground truth EF squared

    train_loss = 0.0
    with torch.set_grad_enabled(phase== "train"):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (video,[filename, ejection]) in dataloader:
                if video.ndim == 6:                      # (B, clips, C, T, H, W)
                    B, clips, C, T, H, W = video.shape
                    video = video.view(B * clips, C, T, H, W)

                    # ejection'i de aynı oranda çoğalt
                    ejection = ejection.view(-1)            # (B,)
                    ejection = ejection.repeat_interleave(clips)  # (B*clips,)
                else:                                    # (B, C, T, H, W) zaten doğru
                    video = video
                    clips = 1

               

                video, ejection = video.float().to(device), ejection.float().to(device)
                # video           = video.permute(0, 2, 1, 3, 4)  # [Batch, Channel, Depth, Height, Width]

                y.append(ejection.cpu().numpy())


                s1              += ejection.sum()              # Mean * n
                s2              += (ejection ** 2).sum()       # Varience * n

                outputs         = model(video)
                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss            = torch.nn.functional.mse_loss(outputs.view(-1), ejection)
                if phase== "train":
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()


                train_loss  += loss.item() * video.size(0)
                n           += video.size(0)
                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(train_loss / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update(1)

                if (phase== "train"):
                    train_losses.append(train_loss/ n)
                elif phase== "val":
                    val_losses.append(train_loss/ n)
                else:
                    pass

    yhat    = np.concatenate(yhat)
    y       = np.concatenate(y)

    return train_loss / n, yhat, y, filename



if __name__ == "__main__":
    run()
