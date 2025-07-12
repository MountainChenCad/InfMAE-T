# validation.py

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from util.video_detection_dataset import VideoDetectionDataset, detection_collate_fn
import traceback


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class SimpleDetector(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        print("Initializing Faster R-CNN with random weights (no pre-training)...")
        self.detector = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        print("Detector initialized successfully!")

    def forward(self, images, targets=None):
        return self.detector(images, targets)


def evaluate_detection(model, dataloader, device, iou_threshold=0.5):
    # (此函数无需修改，保持原样)
    model.eval()
    total_targets_by_class = {i: 0 for i in range(6)}
    detected_targets_by_class = {i: 0 for i in range(6)}
    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_data in dataloader:
            # ... (代码省略以保持简洁)
            pass
    overall_recall = sum(detected_targets_by_class.values()) / sum(total_targets_by_class.values()) if sum(
        total_targets_by_class.values()) > 0 else 0.0
    class_avg_recalls = {
        i: detected_targets_by_class[i] / total_targets_by_class[i] if total_targets_by_class[i] > 0 else 0.0 for i in
        range(6)}
    print("Evaluation completed.")
    return overall_recall, class_avg_recalls


def train_detection_head(train_dataloader, val_dataloader, device, epochs=1):
    detector = SimpleDetector(num_classes=6).to(device)
    params = [p for p in detector.parameters() if p.requires_grad]

    # --- 修复 1: 降低学习率 ---
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # --- 修复结束 ---

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- 修复 2: 引入 GradScaler 以实现稳定的混合精度训练 ---
    scaler = torch.cuda.amp.GradScaler()
    # --- 修复结束 ---

    print(f"Training detection head for {epochs} epochs...")

    for epoch in range(epochs):
        detector.train()
        epoch_losses = []
        total_batches = len(train_dataloader)
        print(f"--- Detection Training Epoch {epoch} ---")

        for batch_idx, batch_data in enumerate(train_dataloader):
            if batch_data is None: continue

            clips, targets_list = batch_data
            if clips is None or not targets_list: continue

            B, T, C, H, W = clips.shape
            middle_t = T // 2
            frames = clips[:, middle_t].to(device)
            targets = [targets_list[b][middle_t] for b in range(B)]

            valid_frames, valid_targets = [], []
            for i, target in enumerate(targets):
                if len(target['boxes']) > 0:
                    valid_frames.append(frames[i])
                    # 确保 target box 和 label 都被正确地移到 device
                    valid_targets.append(
                        {'boxes': target['boxes'].to(device).float(), 'labels': target['labels'].to(device).long()})

            if not valid_frames: continue

            optimizer.zero_grad()

            # --- 修复 3: 使用 autocast 包裹前向传播 ---
            with torch.cuda.amp.autocast():
                loss_dict = detector(valid_frames, valid_targets)
                losses = sum(loss for loss in loss_dict.values())
            # --- 修复结束 ---

            loss_value = losses.item()
            if not np.isfinite(loss_value):
                print(f"\nWarning: NaN loss detected at batch {batch_idx}. Skipping update.")
                continue

            # 使用 scaler 进行反向传播
            scaler.scale(losses).backward()

            # --- 修复 4: 添加梯度裁剪 ---
            # 在 scaler.step() 之前 unscale 梯度
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            # --- 修复结束 ---

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss_value)

            # 更新进度条
            percent = 100. * (batch_idx + 1) / total_batches
            bar_length = 40
            filled_len = int(bar_length * (batch_idx + 1) // total_batches)
            bar = '█' * filled_len + '-' * (bar_length - filled_len)
            print(
                f'\rProgress: |{bar}| {percent:.1f}% [{batch_idx + 1}/{total_batches}] - Batch Loss: {loss_value:.4f}',
                end='')
            if batch_idx + 1 == total_batches:
                print()

        lr_scheduler.step()
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch} summary - Average Loss: {avg_loss:.4f}")

        if val_dataloader is not None and (epoch + 1) % 2 == 0:
            recall, _ = evaluate_detection(detector, val_dataloader, device)
            print(f"Validation Recall after epoch {epoch}: {recall:.4f}")

    return detector


def run_validation_epoch(infmae_model, data_root, device, clip_length=16,
                         transform_train=None, transform_val=None,
                         det_epochs=1, data_ratio=1.0):
    print("=== Starting Validation Process ===")
    original_training_state = infmae_model.training
    try:
        train_dataset = VideoDetectionDataset(data_root=data_root, scene_folders=['scene_1', 'scene_2', 'scene_3'],
                                              clip_length=clip_length, transform=transform_train, data_ratio=0.3)
        val_dataset = VideoDetectionDataset(data_root=data_root, scene_folders=['contest_1'], clip_length=clip_length,
                                            transform=transform_val, data_ratio=0.5)
        print(f"Detection training dataset: {len(train_dataset)} samples")
        print(f"Detection validation dataset: {len(val_dataset)} samples")
        if len(train_dataset) == 0:
            print("Warning: Training dataset is empty. Skipping validation.")
            return 0.0, {i: 0.0 for i in range(6)}
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4,
                                                   pin_memory=True, drop_last=True, collate_fn=detection_collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4,
                                                 pin_memory=True, drop_last=False, collate_fn=detection_collate_fn)
        detector = train_detection_head(train_loader, val_loader, device, det_epochs)
        print("Running final evaluation...")
        recall, class_recalls = evaluate_detection(detector, val_loader, device)
        return recall, class_recalls
    except Exception as e:
        print(f"Error during validation: {e}")
        traceback.print_exc()
        return 0.0, {i: 0.0 for i in range(6)}
    finally:
        infmae_model.train(original_training_state)
        print("=== Validation Process Completed ===")