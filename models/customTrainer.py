import re
import os
import sys
import socket
import random
import argparse
from tqdm import tqdm
import torch.nn as nn
from timm.layers import trunc_normal_
from torch.nn import functional as F
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.optim import AdamW
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist
from torchvision.transforms.functional import center_crop
from torch import multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict, StateDictOptions
)
from src.train_loss import matting_loss, segmentation_loss
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.CustomMamba import MatMat
from models.train_config import DATA_PATHS
from dataset.imagematte import ImageMatteDataset, ImageMatteAugmentation
from dataset.spd import SuperviselyPersonDataset
from dataset.coco import CocoPanopticDataset, CocoPanopticTrainAugmentation
from dataset.youtubevis import YouTubeVISDataset, YouTubeVISAugmentation
from dataset.videomatte import VideoMatteDataset, VideoMatteTrainAugmentation, VideoMatteValidAugmentation
from dataset.augmentation import TrainFrameSampler, ValidFrameSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def fsdp2_shard_model(model: nn.Module, mp_policy: MixedPrecisionPolicy | None, min_params: int = 1e5):
    # shard larger leaf modules first (reduces peak memory), then shard the root
    for m in model.modules():
        if m is model:
            continue
        # leaf-ish: has its own params, not just a container
        if any(p.requires_grad for p in m.parameters(recurse=False)):
            num = sum(p.numel() for p in m.parameters())
            if num >= min_params:
                fully_shard(m, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    return model


class Trainer(object):
    def __init__(self, rank, world_size, scratch=True, finetune=False):
        self.root = Path(__file__).parents[1].resolve()
        self.scratch = scratch
        self.finetune = finetune
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        try:
            self.parse_args()
            self.init_distributed(rank, world_size)
            self.init_datasets()
            self.init_model()
            self.init_optimizer()
            self.init_writer()
            self.train()
            self.cleanup_ddp()

        except Exception as e:
            print(f"[GPU{rank}] Exception: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            dist.destroy_process_group()
            sys.exit(1)

    def parse_args(self):
        parser = argparse.ArgumentParser()

        # Matting datasets
        parser.add_argument("--dataset", type=str or list[str], default='videomatte',
                            choices=['imagematte', 'videomatte'],
                            required=False)
        parser.add_argument("--img-bg-dataset", type=str or list[str], default='bg20k', choices=['bg20k'],
                            help='Name of the background image dataset', required=False)

        # Training variables
        parser.add_argument('--train-hr', action='store_true', default=False)
        parser.add_argument('--resolution-lr', type=int, default=512, required=False)
        parser.add_argument('--resolution-hr', type=int, default=1024, required=False)
        parser.add_argument('--seq-length-lr', type=int, default=7, required=False)
        parser.add_argument('--seq-length-hr', type=int, default=15, required=False)
        parser.add_argument("--batch-size-per-gpu", type=int, default=1, required=False, help='Batch size per GPU')
        parser.add_argument("--epochs", type=int, default=1, required=False, help='Number of epochs')
        parser.add_argument('--initial-lr', type=float, default=0.0001, required=False, help='Initial learning rate')
        parser.add_argument("--num-workers", type=int, default=4, required=False, help='Number of workers')

        # Tensorboard set-up
        parser.add_argument('--log-dir', type=str, default="Experiments", required=False)
        parser.add_argument('--log-train-loss-interval', type=int, default=1000)
        parser.add_argument('--log-train-images-interval', type=int, default=1000)

        # Checkpoint loading and saving
        parser.add_argument('--checkpoint-restore', type=str)
        parser.add_argument('--checkpoint-dir-name', type=str, default="checkpoint", required=False)
        parser.add_argument('--checkpoint-save-interval', type=int, default=500)

        # Slurm Set-Up
        parser.add_argument("--master-address", type=str, default='localhost', required=False)
        parser.add_argument("--master-port", type=str, default='12355', required=False,
                            help='The port to use for communication with the master node.')
        parser.add_argument('--disable-mixed-precision', action='store_true')

        self.args = parser.parse_args()

    def init_distributed(self, rank, world_size):
        torch.backends.cudnn.benchmark = True
        self.rank = rank
        self.log("Rank: {}".format(self.rank))
        self.world_size = world_size
        self.log("World Size: {}".format(self.world_size))
        os.environ['MASTER_ADDR'] = self.args.master_address
        if 'MASTER_PORT' not in os.environ.keys():
            self.port = str(self._find_free_network_port())
            self.log(f"Using port {self.port}")
            os.environ['MASTER_PORT'] = self.port
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    def init_model(self):
        self.log('Initializing model')

        # Build on current CUDA rank
        self.network = MatMat(
            patch_size=2, feat_size=[48, 96, 192, 384, 768],
            deep_supervision=False, use_checkpoint=True
        ).to(self.rank)

        # weight init before sharding is fine (also ok after to_empty+reset if you migrate to meta init)
        self.network.apply(self.init_weights)

        # SyncBN still fine; do this before sharding so the wrapped modules include the converted BN
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)

        mp_policy = None

        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        # Shard submodules + root. You can tweak min_params for your model depth/size.
        self.model = fsdp2_shard_model(self.model, mp_policy=mp_policy, min_params=int(1e5))

        # Restore checkpoint if provided (FSDP2 prefers DCP APIs; see load below)
        if self.args.checkpoint_restore:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint_restore}')
            self._fsdp_load_checkpoint(self.args.checkpoint_restore)

        self.log('Done')
        self.scaler = GradScaler()

    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            num_experiments = self.set_experiment()
            self.experiment_path = os.path.join(self.args.log_dir, f"experiment_{num_experiments:04d}")
            self.writer = SummaryWriter(os.path.join(self.experiment_path, "tensorboard"))

    def set_experiment(self):
        if not os.path.isdir(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        return len(os.listdir(self.args.log_dir))

    def init_optimizer(self):
        self.initial_lr = self.args.initial_lr
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = 10
        self.early_stop_epoch = 350

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs * len(self.dataloader_lr_train),
                                           eta_min=1e-6)

        self.log(f"Using optimizer {self.optimizer}")
        self.log(f"Using scheduler {self.scheduler}")

    def init_weights(self, module):
        """
        Custom weight initialization for MatMat and submodules.
        Includes special handling for embeddings and positional parameters.
        """
        # Conv layers
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Linear layers
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Norm layers
        elif isinstance(module, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

        # Embedding layers
        elif isinstance(module, nn.Embedding):
            trunc_normal_(module.weight, std=0.02)

        # Handle parameters that are direct attributes, like pos_embed
        for name, param in module.named_parameters(recurse=False):
            if "pos_embed" in name:
                trunc_normal_(param, std=0.02)

    def random_crop(self, *imgs):

        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(h // 2, h))
        results = []

        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results

    def compute_deep_supervision_loss(self, preds, true_pha, loss_fn, weights=None):
        """
        pred: list of [B, F, C, H_i, W_i]
        true_pha: [B, F, C, H, W]
        true_fgr: [B, F, C, H, W]
        weights: list of scalars for each scale
        """
        if weights is None:
            weights = [1.0] * len(preds)

        total_loss = 0.0
        for w, pred in zip(weights, preds):
            # Resize GT to match pred
            gt_pha_resized = F.interpolate(
                true_pha.view(-1, *true_pha.shape[2:]),
                size=pred.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).view_as(pred)  # back to [B, F, C, H_i, W_i]
            total_loss += w * loss_fn(pred, gt_pha_resized)['total']

        return total_loss

    def load_next_mat_hr_sample(self):
        try:
            sample = next(self.dataiterator_mat_hr)
        except:
            self.datasampler_hr_train.set_epoch(self.datasampler_hr_train.epoch + 1)
            self.dataiterator_mat_hr = iter(self.dataloader_hr_train)
            sample = next(self.dataiterator_mat_hr)
        return sample

    def load_next_seg_video_sample(self):
        try:
            sample = next(self.dataiterator_seg_video)
        except:
            self.datasampler_seg_video.set_epoch(self.datasampler_seg_video.epoch + 1)
            self.dataiterator_seg_video = iter(self.dataloader_seg_video)
            sample = next(self.dataiterator_seg_video)
        return sample

    def load_next_seg_image_sample(self):
        try:
            sample = next(self.dataiterator_seg_image)
        except:
            self.datasampler_seg_image.set_epoch(self.datasampler_seg_image.epoch + 1)
            self.dataiterator_seg_image = iter(self.dataloader_seg_image)
            sample = next(self.dataiterator_seg_image)
        return sample

    def train(self):
        self.log("Training started")
        for epoch in range(0, self.args.epochs):
            self.epoch = epoch
            self.step = epoch * len(self.dataloader_lr_train)
            self.log(f'Training epoch: {epoch}')
            self.datasampler_lr_train.set_epoch(self.epoch)
            if self.args.train_hr:
                self.datasampler_hr_train.set_epoch(self.epoch)
            self.datasampler_seg_image.set_epoch(self.epoch)
            self.datasampler_seg_video.set_epoch(epoch)

            # Create tensors to hold cumulative loss and count on the current device.
            self.train_lr_loss = torch.tensor(0.0, device=self.rank)
            self.train_lr_count = torch.tensor(0, device=self.rank)

            if self.args.train_hr:
                self.train_hr_loss = torch.tensor(0.0, device=self.rank)
                self.train_hr_count = torch.tensor(0, device=self.rank)

            for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_lr_train, disable=False, dynamic_ncols=True):
                self.train_mat(true_fgr, true_pha, true_bgr, tag='lr')

                # High resolution pass
                if self.args.train_hr:
                    true_fgr, true_pha, true_bgr = self.load_next_mat_hr_sample()
                    self.train_mat(true_fgr, true_pha, true_bgr, tag='hd')

                # Segmentation pass
                if self.step % 2 == 0:
                    true_img, true_seg = self.load_next_seg_video_sample()
                    self.train_seg(true_img, true_seg, log_label='seg_video')
                else:
                    true_img, true_seg = self.load_next_seg_image_sample()
                    self.train_seg(true_img.unsqueeze(1), true_seg.unsqueeze(1), log_label='seg_image')

                self.step += 1
            # Aggregate results from all ranks.
            dist.all_reduce(self.train_lr_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.train_lr_count, op=dist.ReduceOp.SUM)
            avg_lr_loss = self.train_lr_loss.item() / self.train_lr_count.item()
            if self.args.train_hr:
                dist.all_reduce(self.train_hr_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.train_hr_count, op=dist.ReduceOp.SUM)
                avg_hr_loss = self.train_hr_loss.item() / self.train_hr_count.item()
                self.log(f'Epoch HR average loss: {avg_hr_loss}')

            val_loss = self.validate()

            if self.rank == 0 and torch.no_grad():
                self.log(f'Epoch LR average loss: {avg_lr_loss}')
                self.log(f'Epoch validation loss: {val_loss}')

                self.writer.add_scalar('Loss/train_lr', avg_lr_loss, self.epoch)
                if self.args.train_hr:
                    self.writer.add_scalar('Loss/train_hr', avg_hr_loss, self.epoch)
                self.writer.add_scalar('Loss/val', val_loss, self.epoch)

    def train_mat(self, true_fgr, true_pha, true_bgr, tag):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        # true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        B, F, C, H, W = true_src.shape

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not self.args.disable_mixed_precision):
            self.log(("Before forward pass:", torch.cuda.memory_allocated() / 1e9, "GB"))
            outs = self.model(true_src)
            self.log(("After forward pass:", torch.cuda.memory_allocated() / 1e9, "GB"))
            loss = matting_loss(pred_fgr=outs[0], pred_pha=outs[1],
                                true_pha=true_pha, true_fgr=true_fgr)
            if tag == 'lr':
                self.train_lr_loss += loss['total'].item() * B
                self.train_lr_count += B
            elif tag == 'hd':
                self.train_hr_loss += loss['total'].item() * B
                self.train_hr_count += B

        self.scaler.scale(loss['total']).backward()
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
                self.writer.add_scalar(f'train_{tag}_loss', loss['total'].item(), self.step)

            if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
                grid_a = make_grid(outs[1].flatten(0, 1).detach().cpu(), nrow=outs[1].size(1))
                self.writer.add_image(f'train_{tag}_pred_pha', grid_a, self.step)
                # grid_b = make_grid(outs[0].flatten(0, 1).detach().cpu(), nrow=outs[0].size(1))
                # self.writer.add_image(f'train_{tag}_pred_fgr', grid_b, self.step)
                # grid_c = make_grid(true_fgr.flatten(0, 1).detach().cpu(), nrow=true_fgr.size(1))
                # self.writer.add_image(f'train_{tag}_true_fgr', grid_c, self.step)
                grid_d = make_grid(true_pha.flatten(0, 1).detach().cpu(), nrow=true_pha.size(1))
                self.writer.add_image(f'train_{tag}_true_pha', grid_d, self.step)
                grid_f = make_grid(true_src.flatten(0, 1).detach().cpu(), nrow=true_src.size(1))
                self.writer.add_image(f'train_{tag}_true_src', grid_f, self.step)
                torch.cuda.empty_cache()

    def train_seg(self, true_img, true_seg, log_label):
        true_img = true_img.to(self.rank, non_blocking=True)
        true_seg = true_seg.to(self.rank, non_blocking=True)
        # true_img, true_seg = self.random_crop(true_img, true_seg)

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not self.args.disable_mixed_precision):
            pred_seg = self.model(true_img, seg_pass=True)
            loss = segmentation_loss(pred_seg, true_seg)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_loss_interval == 0:
                self.writer.add_scalar(f'{log_label}_loss', loss.item(), self.step)

            if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_images_interval == 0:
                self.writer.add_image(f'{log_label}_pred_seg',
                                      make_grid(pred_seg.flatten(0, 1).detach().cpu(), nrow=self.args.seq_length_lr),
                                      self.step)
                self.writer.add_image(f'{log_label}_true_seg',
                                      make_grid(true_seg.flatten(0, 1).detach().cpu(), nrow=self.args.seq_length_lr),
                                      self.step)
                self.writer.add_image(f'{log_label}_true_img',
                                      make_grid(true_img.flatten(0, 1).detach().cpu(), nrow=self.args.seq_length_lr),
                                      self.step)
                torch.cuda.empty_cache()

    def validate(self):
        self.log(f'Validating epoch: {self.epoch}')
        self.model.eval()
        self.datasampler_valid.set_epoch(self.epoch)
        # Create tensors to hold cumulative loss and count on the current device.
        total_loss = torch.tensor(0.0, device=self.rank)
        total_count = torch.tensor(0, device=self.rank)

        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=not self.args.disable_mixed_precision):
                for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid, dynamic_ncols=True):
                    true_fgr = true_fgr.to(self.rank, non_blocking=True)
                    true_pha = true_pha.to(self.rank, non_blocking=True)
                    true_bgr = true_bgr.to(self.rank, non_blocking=True)
                    true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                    batch_size = true_src.size(0)
                    outs = self.model(true_src)
                    val_loss = matting_loss(pred_fgr=outs[0], pred_pha=outs[1], true_pha=true_pha, true_fgr=true_fgr)
                    total_loss += val_loss.item() * batch_size
                    total_count += batch_size

            if self.rank == 0:
                self.writer.add_image(f'val_pred_pha',
                                      make_grid(outs[0].flatten(0, 1).detach().cpu(), nrow=outs[0].size(1)), self.step)
                # self.writer.add_image(f'val_true_fgr', make_grid(true_fgr.flatten(0, 1).detach().cpu(), nrow=true_fgr.size(1)), self.step)
                self.writer.add_image(f'val_true_pha',
                                      make_grid(true_pha.flatten(0, 1).detach().cpu(), nrow=true_pha.size(1)),
                                      self.step)
                self.writer.add_image(f'val_true_src',
                                      make_grid(true_src.flatten(0, 1).detach().cpu(), nrow=true_src.size(1)),
                                      self.step)
                torch.cuda.empty_cache()

        # Aggregate results from all ranks.
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        av_val_loss = total_loss.item() / total_count.item()

        if self.rank == 0:
            self.log(f'Validation average loss: {av_val_loss}')
            self.writer.add_scalar('valid_loss', av_val_loss, self.step)

        if self.rank == 0:
            # Check if this is the best loss and save the model
            if not hasattr(self, 'best_loss') or av_val_loss < self.best_loss:
                self.best_loss = av_val_loss
                best_model = 'best.pth'
                checkpoint_path = os.path.join(self.experiment_path, 'checkpoint')
                self.log(f'New best model (val {av_val_loss:.6f}). Saving FSDP2 checkpoint...')
                self._fsdp_save_checkpoint(checkpoint_path)

        self.model.train()
        dist.barrier()

        return av_val_loss

    def _fsdp_save_checkpoint(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        model_state_dict = get_model_state_dict(
            model=self.model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if dist.get_rank() == 0:
            torch.save(model_state_dict, os.path.join(ckpt_dir, "model_state_dict.pt"))

    def _fsdp_load_checkpoint(self, ckpt_path: str):
        if dist.get_rank() == 0:
            full_sd = torch.load(ckpt_path, mmap=True, weights_only=True, map_location='cpu')
        else:
            full_sd = None
        set_model_state_dict(
            model=self.model,
            model_state_dict=full_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        dist.barrier()

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
        # Matting Datasets size = H, W
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr // 2, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr // 2, self.args.resolution_lr)

        # Matting datasets:
        if self.args.dataset == 'videomatte':
            self.dataset_lr_train = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                resolution=size_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation(size_lr))

            if self.args.train_hr:
                self.dataset_hr_train = VideoMatteDataset(
                    videomatte_dir=DATA_PATHS['videomatteHD']['train'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    resolution=size_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=VideoMatteTrainAugmentation(size_hr))

            self.dataset_valid = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['valid'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                resolution=size_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=VideoMatteValidAugmentation(size_hr if self.args.train_hr else size_lr))

        elif self.args.dataset == 'am2k':
            self.dataset_lr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['am2k']['train'],
                background_image_dir=DATA_PATHS['bg20k']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                resolution=size_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_lr))

            if self.args.train_hr:
                self.dataset_hr_train = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['am2k']['train'],
                    background_image_dir=DATA_PATHS['bg20k']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    resolution=size_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr))

            self.dataset_valid = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['am2k']['valid'],
                background_image_dir=DATA_PATHS['bg20k']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                resolution=size_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=ImageMatteAugmentation(size_hr if self.args.train_hr else size_lr))

        elif self.args.dataset == 'imagematte':
            self.dataset_lr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['train'],
                background_image_dir=DATA_PATHS['bg20k']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                resolution=size_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_lr))

            if self.args.train_hr:
                self.dataset_hr_train = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imagematte']['train'],
                    background_image_dir=DATA_PATHS['bg20k']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    resolution=size_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr))

                self.dataset_valid = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imagematte']['valid'],
                    background_image_dir=DATA_PATHS['bg20k']['valid'],
                    background_video_dir=DATA_PATHS['background_videos']['valid'],
                    size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                    resolution=size_hr if self.args.train_hr else self.args.resolution_lr,
                    seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                    seq_sampler=ValidFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr if self.args.train_hr else size_lr))

        elif self.args.dataset == 'imgbra':
            self.dataset_lr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imgbra']['train'],
                background_image_dir=DATA_PATHS['brainstorm_bg_images']['train'],
                background_video_dir=DATA_PATHS['brainstorm_bgs']['train'],
                size=self.args.resolution_lr,
                resolution=size_hr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_lr))

            if self.args.train_hr:
                self.dataset_hr_train = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imgbra']['train'],
                    background_image_dir=DATA_PATHS['brainstorm_bg_images']['train'],
                    background_video_dir=DATA_PATHS['brainstorm_bgs']['train'],
                    size=self.args.resolution_hr,
                    resolution=size_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr))

                self.dataset_valid = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imgbra']['valid'],
                    background_image_dir=DATA_PATHS['brainstorm_bg_images']['valid'],
                    background_video_dir=DATA_PATHS['brainstorm_bgs']['valid'],
                    size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                    resolution=size_hr if self.args.train_hr else self.args.resolution_lr,
                    seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                    seq_sampler=ValidFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr if self.args.train_hr else size_lr))

        # Matting dataloaders:
        self.datasampler_lr_train = DistributedSampler(
            dataset=self.dataset_lr_train,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_lr_train = DataLoader(
            dataset=self.dataset_lr_train,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_lr_train,
            pin_memory=True,
            drop_last=True)
        if self.args.train_hr:
            self.datasampler_hr_train = DistributedSampler(
                dataset=self.dataset_hr_train,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True)
            self.dataloader_hr_train = DataLoader(
                dataset=self.dataset_hr_train,
                batch_size=self.args.batch_size_per_gpu,
                num_workers=self.args.num_workers,
                sampler=self.datasampler_hr_train,
                pin_memory=True,
                drop_last=True)
        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True)

        # Segementation datasets
        self.log('Initializing image segmentation datasets')
        self.dataset_seg_image = ConcatDataset([
            CocoPanopticDataset(
                imgdir=DATA_PATHS['coco_panoptic']['imgdir'],
                anndir=DATA_PATHS['coco_panoptic']['anndir'],
                annfile=DATA_PATHS['coco_panoptic']['annfile'],
                resolution=size_lr,
                transform=CocoPanopticTrainAugmentation(size_lr)),
            SuperviselyPersonDataset(
                imgdir=DATA_PATHS['spd']['imgdir'],
                segdir=DATA_PATHS['spd']['segdir'],
                resolution=size_lr,
                transform=CocoPanopticTrainAugmentation(size_lr))
        ])

        self.datasampler_seg_image = DistributedSampler(
            dataset=self.dataset_seg_image,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)

        self.dataloader_seg_image = DataLoader(
            dataset=self.dataset_seg_image,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_image,
            pin_memory=True)

        # Datasampler and Dataloader for Validation
        self.datasampler_valid = DistributedSampler(
            dataset=self.dataset_valid,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=False)

        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            sampler=self.datasampler_valid,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True)

        self.log('Initializing video segmentation datasets')
        self.dataset_seg_video = YouTubeVISDataset(
            videodir=DATA_PATHS['youtubevis']['videodir'],
            annfile=DATA_PATHS['youtubevis']['annfile'],
            size=self.args.resolution_lr,
            resolution=size_lr,
            seq_length=self.args.seq_length_lr,
            seq_sampler=TrainFrameSampler(speed=[1]),
            transform=YouTubeVISAugmentation(size_lr))

        self.datasampler_seg_video = DistributedSampler(
            dataset=self.dataset_seg_video,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)

        self.dataloader_seg_video = DataLoader(
            dataset=self.dataset_seg_video,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_video,
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