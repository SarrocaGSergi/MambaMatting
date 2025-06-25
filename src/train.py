import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
from torch import optim
import torchvision.transforms.v2 as T
import torchvision.transforms as t

from dataset.imagematte import ImageMatteDataset
from dataset.imgmatte import ImageMatte
from torchvision import transforms
from models.swing_mamba import SwinUMamba, matting_loss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from assets import define_experiment
from pathlib import Path

_PROJECT_ROOT_ = Path(__file__).parents[1].resolve()
torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 42
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(f"DEVICE: {device}")
print("Load SwingUMamba")

model = SwinUMamba(in_chans=3, out_chans=1, feat_size=[48, 96, 192, 384, 768], deep_supervision=True,
                   hidden_size=768).to('cuda')
# model = load_pretrained_ckpt(model)
model.load_state_dict(torch.load(os.path.join(_PROJECT_ROOT_, "upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth"),
                                 weights_only=True, map_location=device), strict=False)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))

print("Loaded")

size_hr = (512, 288)
TRANSFORM = T.Compose([
    T.Resize(size_hr),
    T.CenterCrop(size_hr),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.7),
    T.RandomRotation(degrees=(15, 15)),
    T.RandomAffine(degrees=0.1),
    t.ToTensor(),
])

dataset_train = ImageMatteDataset(imagematte_dir="/mnt/nvme0n1/Datasets/matting-data/ImageMatte/train",
                                  background_image_dir= "/mnt/nvme0n1/Datasets/matting-data/ImageMatte/train/bgr",
                                  transform=TRANSFORM, size=(512,288), seq_length=1)
dataset_val = ImageMatteDataset(imagematte_dir="/mnt/nvme0n1/Datasets/matting-data/ImageMatte/val",
                                background_image_dir= "/mnt/nvme0n1/Datasets/matting-data/ImageMatte/val/bgr",
                                transform=TRANSFORM, size=(512,288), seq_length=1)

dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=True, drop_last=True)

opt = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.5, 0.9), weight_decay=10e-5)
scheduler = CosineAnnealingWarmRestarts(opt, T_0=1000, T_mult=1, eta_min=1e-6)
# Plotting on Tensorboard
print("Creating summary writer...")
experiment_dir = define_experiment()
logs_dir = os.path.join(experiment_dir, "tensorboard")
writer = SummaryWriter(logs_dir)
print("Summary Created on: " + logs_dir)

# Training Loop
# Lists to keep track of progress
ITERS = 0

print("Starting Training Loop...")
# For each epoch
num_epochs = 1
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader_train, 0):
        model.train()
        output = []
        true_fgr = data[0]
        true_bgr = data[2]
        true_pha = data[1]
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        # Split into smaller batches of 4
        # for j in range(0, 16, 4):
        #     true_src_batch = data[0][j:j + 4]
        #     # Format batch
        #     true_src = true_src_batch.to(device)
        #
        #     output.append(small_pred[0].cpu())
        output = model(true_src.to(device))
        output = torch.cat(output, dim=0)
        print(output.requires_grad)
        loss = matting_loss(output, true_pha)
        loss['total'].backward()
        scheduler.step()
        opt.step()
        print(loss['total'].item())

        # Output training stats
        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tPha_L1: %.4f\tPha_Laplacian: %.4f\tMatting_Loss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader_train), loss['pha_l1'], loss['pha_laplacian'], loss['total']))

            writer.add_scalar('L1 Loss', loss['pha_l1'], global_step=ITERS)
            writer.add_scalar('Laplacian Loss', loss['pha_laplacian'], global_step=ITERS)
            writer.add_scalar('Matting Loss', loss['total'], global_step=ITERS)

    for idx, data in enumerate(dataloader_val, 0):
        ## Train with all-real batch

        # Format batch
        true_fgr = data[0].to(device)
        true_bgr = data[2].to(device)
        true_pha = data[1].to(device)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        pred_pha = model(true_src)
        val_loss = matting_loss(pred_pha[0], true_pha)

        # Output training stats
        if idx % 500 == 0:
            print('[%d/%d][%d/%d]\tPha_L1: %.4f\tPha_Laplacian: %.4f\tMatting_Loss: %.4f'
                  % (epoch, num_epochs, idx, len(dataloader_val), val_loss['pha_l1'], val_loss['pha_laplacian'],
                     val_loss['total']))

            writer.add_scalar('L1 Val Loss', val_loss['pha_l1'], global_step=ITERS)
            writer.add_scalar('Laplacian Val Loss', val_loss['pha_laplacian'], global_step=ITERS)
            writer.add_scalar('Matting Val Loss', val_loss['total'], global_step=ITERS)
        ITERS += 1

    print('[%d/%d][%d/%d]\tTrain Total: %.4f\tValidation Total: %.4f'
          % (epoch, num_epochs, i, len(dataloader_train), loss['total'].item(), val_loss['total'].item()))

writer.close()