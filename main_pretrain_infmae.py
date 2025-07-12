# main_pretrain_infmae.py

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import timm

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.video_pretrain_dataset import VideoMAEPretrainDataset
from util.video_detection_dataset import VideoDetectionDataset, detection_collate_fn
import models_infmae_skip4
from engine_pretrain import train_one_epoch
from validation import run_validation_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('Infrared VideoMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='infmae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio for video tubes (higher is better for video)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Video specific parameters
    parser.add_argument('--clip_length', type=int, default=16, help='Number of frames in each video clip')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../InfAIM/dataset', type=str,
                        help='Root path to dataset (containing scene_1, scene_2, etc.)')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='Ratio of training data to use (0.0-1.0)')

    # Validation parameters
    parser.add_argument('--validate', action='store_true', help='Run validation')
    parser.add_argument('--val_epochs', type=int, default=20, help='Validation frequency')
    parser.add_argument('--det_epochs', type=int, default=10, help='Detection head training epochs')

    parser.add_argument('--output_dir', default='./output_inf_videomae', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_inf_videomae', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix random seeds
    global_rank = misc.get_rank()
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.5, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.425, 0.425, 0.425], std=[0.200, 0.200, 0.200])])

    transform_val = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.425, 0.425, 0.425], std=[0.200, 0.200, 0.200])])

    # Training dataset
    train_scene_folders = ['scene_1', 'scene_2', 'scene_3']
    dataset_train = VideoMAEPretrainDataset(
        data_root=args.data_path,
        scene_folders=train_scene_folders,
        clip_length=args.clip_length,
        transform=transform_train,
        data_ratio=args.data_ratio
    )
    print(f"Training dataset: {len(dataset_train)} samples")

    # Data loaders
    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Model
    model = models_infmae_skip4.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        clip_length=args.clip_length
    )
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Optimizer
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Load checkpoint if resuming
    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_recall = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 确保模型在训练模式
        model.train()

        # Validation (在训练之前进行，避免影响训练)
        val_stats = {}
        if args.validate and epoch % args.val_epochs == 0 and epoch > 0:
            print("Running validation...")

            # 运行独立的验证过程
            recall, class_recalls = run_validation_epoch(
                model_without_ddp, args.data_path, device, args.clip_length,
                transform_train, transform_val, args.det_epochs, args.data_ratio
            )

            val_stats = {
                'val_recall': recall,
                'val_class_recalls': class_recalls
            }

            print(f"Validation Results - Overall Recall: {recall:.4f}")
            for class_id, class_recall in class_recalls.items():
                class_names = ["drone", "car", "ship", "bus", "pedestrian", "cyclist"]
                print(f"  {class_names[class_id]}: {class_recall:.4f}")

            # Save best model
            if recall > best_recall:
                best_recall = recall
                if args.output_dir:
                    misc.save_model(
                        args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler,
                        filename='best_checkpoint.pth')

        # 确保模型在训练模式
        model.train()

        # 训练一个epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # *** 修复：每个epoch都保存checkpoint ***
        if args.output_dir:
            misc.save_model(
                args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler)

            # 额外保存最近的几个epoch
            if epoch >= args.epochs - 5:  # 保存最后5个epoch
                misc.save_model(
                    args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler,
                    filename=f'checkpoint-last-{epoch}.pth')

        # Logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)