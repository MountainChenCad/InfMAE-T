# engine_pretraining.py

import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # --- 修复：添加进度条 ---
    total_batches = len(data_loader)
    # --- 修复结束 ---

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # --- 修复：修改循环以手动控制打印 ---
    for data_iter_step, (samples, _) in enumerate(data_loader):
        # --- 修复结束 ---
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # --- 修复：打印实时进度条 ---
        percent = 100. * (data_iter_step + 1) / total_batches
        bar_length = 40
        filled_len = int(bar_length * (data_iter_step + 1) // total_batches)
        bar = '█' * filled_len + '-' * (bar_length - filled_len)

        avg_loss = metric_logger.loss.avg

        # 使用 \r 实现单行刷新
        print(
            f'\r{header} |{bar}| {percent:.1f}% [{data_iter_step + 1}/{total_batches}] - Avg Loss: {avg_loss:.4f} - LR: {lr:.6f}',
            end='')
        if data_iter_step + 1 == total_batches:
            print()  # 在 epoch 结束时换行
        # --- 修复结束 ---

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
