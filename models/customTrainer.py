import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import torch
import socket
from torchvision.utils import make_grid
from dataset.imagematte import ImageMatteDataset

from torch.utils.tensorboard import SummaryWriter
from train_config import DATA_PATHS
from SwinUamba import SwinUMamba
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from src.train_loss import matting_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path


class Trainer(object):
    def __init__(self, rank, world_size, scratch=True, finetune=False):
        self.root = Path(__file__).parents[1].resolve()
        self.scratch = scratch
        self.finetune = finetune
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_optimizer()
        self.init_writer()
        self.train()
        self.cleanup_ddp()

    def init_distributed(self, rank, world_size):
        torch.backends.cudnn.benchmark = True
        self.rank = rank
        self.log("Rank: {}".format(self.rank))
        self.world_size = world_size
        self.log("World Size: {}".format(self.world_size))
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            # self.port = str(self._find_free_network_port())
            # self.log(f"Using port {self.port}")
            os.environ['MASTER_PORT'] = str(12789)
        # str(port)
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    def init_model(self):
        self.log('Initializing model')
        self.network = SwinUMamba(in_chans=3, out_chans=1, feat_size=[48, 96, 192, 384, 768], deep_supervision=True,
                                  hidden_size=768).to(self.rank)
        self._load_ckpt()
        self.log("Parallelizing model")
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        self.log('Done')
        self.scaler = GradScaler()

    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter('Experiments/log_dir/imagematte/')

    def init_optimizer(self):
        self.num_epochs = 50
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = 10
        self.early_stop_epoch = 350

        self.optimizer = AdamW(
            self.model_ddp.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6, verbose=False)
        self.log(f"Using optimizer {self.optimizer}")
        self.log(f"Using scheduler {self.scheduler}")

    def train(self):
        self.log("Training started")
        self.iters = 0
        for epoch in range(0, self.num_epochs):
            self.epoch = epoch
            self.step = epoch * len(self.dataloader_lr_train)
            self.log(f'Training epoch: {epoch}')
            # Create tensors to hold cumulative loss and count on the current device.
            total_loss = torch.tensor(0.0, device=self.rank)
            total_count = torch.tensor(0, device=self.rank)

            for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_lr_train, disable=False, dynamic_ncols=True):
                true_fgr = true_fgr.to(self.rank, non_blocking=True)
                true_pha = true_pha.to(self.rank, non_blocking=True)
                true_bgr = true_bgr.to(self.rank, non_blocking=True)
                # true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
                true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                batch_size = true_src.size(0)

                with autocast(enabled=True):
                    pred_pha = self.model_ddp(true_src)
                    loss = matting_loss(pred_pha[0], true_pha)
                    total_loss += matting_loss(pred_pha[0], true_pha)['total'].item() * batch_size
                    total_count += batch_size

                self.scaler.scale(loss['total']).backward()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

                self.iters += 1
                if self.rank == 0 and self.iters % 50 == 0:
                    for loss_name, loss_value in loss.items():
                        self.writer.add_scalar(f'train_{loss_name}', loss_value, self.iters)

                if self.rank == 0 and self.iters % 50 == 0:
                    self.writer.add_image(f'train_pred_pha',
                                          make_grid(pred_pha[0], nrow=pred_pha[0].size(0), padding=2), self.iters)
                    self.writer.add_image(f'train_true_pha', make_grid(true_pha, nrow=true_pha.size(0), padding=2),
                                          self.iters)
                    self.writer.add_image(f'train_true_src', make_grid(true_src, nrow=true_src.size(0), padding=2),
                                          self.iters)

            # Aggregate results from all ranks.
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            avg_loss = total_loss.item() / total_count.item()

            if self.rank == 0:
                self.log(f'Epoch average loss: {avg_loss}')
                self.writer.add_scalar('Epoch_loss', avg_loss, self.step)

            self.validate()

    def validate(self):
        self.log(f'Validating epoch: {self.epoch}')
        self.model_ddp.eval()
        self.datasampler_valid.set_epoch(self.epoch)
        # Create tensors to hold cumulative loss and count on the current device.
        total_loss = torch.tensor(0.0, device=self.rank)
        total_count = torch.tensor(0, device=self.rank)

        with torch.no_grad():
            with autocast(enabled=False):
                for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid, dynamic_ncols=True):
                    true_fgr = true_fgr.to(self.rank, non_blocking=True)
                    true_pha = true_pha.to(self.rank, non_blocking=True)
                    true_bgr = true_bgr.to(self.rank, non_blocking=True)
                    true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                    batch_size = true_src.size(0)
                    pred_pha = self.model_ddp(true_src)
                    total_loss += matting_loss(pred_pha[0], true_pha)['total'].item() * batch_size
                    total_count += batch_size

            if self.rank == 0:
                self.writer.add_image(f'val_pred_pha', make_grid(pred_pha[0], nrow=pred_pha[0].size(0), padding=2),
                                      self.iters)
                self.writer.add_image(f'val_true_pha', make_grid(true_pha, nrow=true_pha.size(0), padding=2),
                                      self.iters)
                self.writer.add_image(f'val_true_src', make_grid(true_src, nrow=true_src.size(0), padding=2),
                                      self.iters)

        # Aggregate results from all ranks.
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / total_count.item()

        if self.rank == 0:
            self.log(f'Validation average loss: {avg_loss}')
            self.writer.add_scalar('valid_loss', avg_loss, self.step)

        # if self.rank == 0:
        #     # Check if this is the best loss and save the model
        #     if not hasattr(self, 'best_loss') or avg_loss < self.best_loss:
        #         self.best_loss = avg_loss
        #         best_epoch_path = os.path.join(self.args.checkpoint_dir, 'best_epoch')
        #         best_model = 'best.pth'
        #         if not os.path.exists(best_epoch_path):
        #             os.makedirs(best_epoch_path, exist_ok=True)
        #         best_epoch = os.path.join(best_epoch_path,  f'best_{self.epoch}.pth')
        #         if not os.path.exists(self.args.checkpoint_dir):
        #             os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        #         torch.save(self.model.state_dict(), os.path.join(best_epoch))
        #         torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_dir, best_model))
        #         self.log(f'New best epoch saved at: {best_epoch}')
        #         self.log(f'New best model saved at: {os.path.join(self.args.checkpoint_dir, best_model)}')

        self.model_ddp.train()
        dist.barrier()

    def _load_ckpt(self):
        ckpt_path = os.path.join(self.root, "vssmtiny_dp01_ckpt_epoch_292.pth")
        self.log(f"Loading weights from: {ckpt_path}")
        skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias",
                       "patch_embed.proj.weight", "patch_embed.proj.bias",
                       "patch_embed.norm.weight", "patch_embed.norm.weight"]

        ckpt = torch.load(ckpt_path, map_location=f'cuda:{self.rank}', weights_only=False)
        model_dict = self.network.state_dict()
        for k, v in ckpt['model'].items():
            if k in skip_params:
                self.log(f"Skipping weights: {k}")
                continue
            kr = f"vssm_encoder.{k}"
            if "downsample" in kr:
                i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
                kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
                assert kr in model_dict.keys()
            if kr in model_dict.keys():
                assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
                model_dict[kr] = v
            else:
                self.log(f"Passing weights: {k}")

        self.network.load_state_dict(model_dict)

        return self.network

    def _find_free_network_port(self) -> int:
        """Finds a free port on localhost.

        It is useful in single-node training when we don't want to connect to a real main node but have to set the
        `MASTER_PORT` environment variable.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def cleanup_ddp(self):
        dist.destroy_process_group()

    def log(self, msg):
        if self.rank == 0:
            print(f'[GPU{self.rank}] {msg}')

    def init_datasets(self):
        size_lr = (1920, 1080)

        self.dataset_lr_train = ImageMatteDataset(
            imagematte_dir=DATA_PATHS['imagematte']['train'],
            background_image_dir=DATA_PATHS['bg20k']['train'],
            size=size_lr[0],
            transform=None)

        self.dataset_valid = ImageMatteDataset(
            imagematte_dir=DATA_PATHS['imagematte']['valid'],
            background_image_dir=DATA_PATHS['background_images']['valid'],
            size=size_lr[0],
            transform=None)

        # Matting dataloaders:
        self.datasampler_lr_train = DistributedSampler(
            dataset=self.dataset_lr_train,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)

        self.dataloader_lr_train = DataLoader(
            dataset=self.dataset_lr_train,
            batch_size=8,
            num_workers=8,
            sampler=self.datasampler_lr_train,
            pin_memory=True)

        # Datasamplers and Dataloader for Validation
        self.datasampler_valid = DistributedSampler(
            dataset=self.dataset_valid,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=False)

        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            sampler=self.datasampler_valid,
            batch_size=8,
            num_workers=8,
            pin_memory=True,
            drop_last=True)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)