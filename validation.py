# validation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
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


class InfMAEFeatureExtractor(nn.Module):
    """使用InfMAE作为特征提取器，处理动态尺寸输入"""

    def __init__(self, infmae_model):
        super().__init__()
        self.infmae = infmae_model
        self.infmae.eval()  # 冻结InfMAE参数

        # 冻结InfMAE参数
        for param in self.infmae.parameters():
            param.requires_grad = False

        # InfMAE的输入尺寸
        self.infmae_input_size = 224

        # 获取InfMAE的特征维度（需要根据实际模型调整）
        # 通常ViT-Base是768，ViT-Large是1024
        self.feature_dim = 768  # 根据你的InfMAE配置调整

        print(f"InfMAE feature extractor initialized (input size: {self.infmae_input_size})")

    def forward(self, x):
        # x: [B, C, H, W] - 输入图像，尺寸可能各异
        B, C, original_H, original_W = x.shape

        # 将图像resize到InfMAE期望的尺寸
        x_resized = F.interpolate(
            x,
            size=(self.infmae_input_size, self.infmae_input_size),
            mode='bilinear',
            align_corners=False
        )

        # 为InfMAE准备输入：添加时间维度
        x_video = x_resized.unsqueeze(1)  # [B, 1, C, H, W]

        with torch.no_grad():
            # 获取InfMAE的特征
            features = self.infmae.forward_encoder(x_video, mask_ratio=0.0)
            # features: [B, N, D] 其中N是patch数量，D是特征维度

            # 计算patch的空间布局
            patch_size = 16  # InfMAE通常使用16x16的patch
            num_patches_per_dim = self.infmae_input_size // patch_size  # 224//16 = 14

            # 重新组织为空间特征图
            spatial_features = features.view(B, num_patches_per_dim, num_patches_per_dim, self.feature_dim)
            spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, D, 14, 14]

        # 将特征上采样到合适的检测尺寸
        # 对于检测，我们通常希望特征图比原图小一些，比如原图的1/4或1/8
        target_H = original_H // 4  # 可以调整这个下采样比例
        target_W = original_W // 4

        upsampled_features = F.interpolate(
            spatial_features,
            size=(target_H, target_W),
            mode='bilinear',
            align_corners=False
        )

        # 返回多尺度特征（FPN格式）
        # 我们创建多个尺度的特征图
        features_dict = {}

        # 原始尺度
        features_dict['0'] = upsampled_features

        # 创建额外的尺度用于FPN
        features_dict['1'] = F.avg_pool2d(upsampled_features, kernel_size=2, stride=2)
        features_dict['2'] = F.avg_pool2d(features_dict['1'], kernel_size=2, stride=2)
        features_dict['3'] = F.avg_pool2d(features_dict['2'], kernel_size=2, stride=2)

        return features_dict


class MultiScaleInfMAEBackbone(nn.Module):
    """多尺度InfMAE backbone，兼容FPN"""

    def __init__(self, infmae_model):
        super().__init__()
        self.feature_extractor = InfMAEFeatureExtractor(infmae_model)

        # 特征维度
        in_channels_list = [768, 768, 768, 768]  # 四个尺度的特征
        out_channels = 256

        # 创建FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        # 设置out_channels属性，供FasterRCNN使用
        self.out_channels = out_channels

    def forward(self, x):
        # 提取多尺度特征
        features = self.feature_extractor(x)

        # 通过FPN
        fpn_features = self.fpn(features)

        return fpn_features


class LightweightDetectionHead(nn.Module):
    """轻量级检测头，使用InfMAE特征"""

    def __init__(self, infmae_model, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

        print("Creating lightweight detection head using InfMAE features...")

        # 创建backbone
        self.backbone = MultiScaleInfMAEBackbone(infmae_model)

        # 创建FasterRCNN的其他组件
        from torchvision.models.detection.faster_rcnn import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
        from torchvision.models.detection.roi_heads import RoIHeads

        # RPN anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # RPN head
        rpn_head = RPNHead(
            self.backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0]
        )

        # ROI heads
        from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
        from torchvision.ops import MultiScaleRoIAlign

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            self.backbone.out_channels * resolution ** 2,
            representation_size
        )

        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes + 1  # +1 for background
        )

        from torchvision.models.detection.roi_heads import RoIHeads
        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05, nms_thresh=0.5, detections_per_img=100
        )

        # 创建完整的检测器
        self.detector = FasterRCNN(
            self.backbone,
            num_classes=None,  # 我们手动设置了roi_heads
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            roi_heads=roi_heads,
        )

        print("Lightweight detection head created successfully!")

    def forward(self, images, targets=None):
        return self.detector(images, targets)


def comprehensive_evaluation(model, dataloader, device, iou_threshold=0.3):
    """全面评估，包括precision, recall, F1和虚警率分析"""
    model.eval()

    all_predictions = []
    all_targets = []

    print(f"\nCollecting predictions and targets (IoU threshold: {iou_threshold})...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
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
                    valid_targets.append(target)
                    valid_frames.append(frames[i])

            if not valid_targets: continue

            try:
                predictions = model(valid_frames)

                for pred, target in zip(predictions, valid_targets):
                    all_predictions.append({
                        'boxes': pred['boxes'].cpu(),
                        'labels': pred['labels'].cpu(),
                        'scores': pred['scores'].cpu()
                    })
                    all_targets.append({
                        'boxes': target['boxes'],
                        'labels': target['labels']
                    })
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

            if batch_idx % 20 == 0:
                print(f"Progress: {batch_idx + 1}/{len(dataloader)}")

    print(f"Collected {len(all_predictions)} prediction-target pairs")

    if len(all_predictions) == 0:
        print("No valid predictions collected!")
        return 0.0, {i: 0.0 for i in range(6)}

    # 分析不同置信度阈值
    confidence_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    print("\n=== COMPREHENSIVE DETECTION ANALYSIS ===")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"Total samples: {len(all_predictions)}")

    results = {}

    for conf_thresh in confidence_thresholds:
        # 统计指标
        true_positives = 0
        false_positives = 0
        total_targets = 0
        total_predictions = 0

        # 按类别统计
        class_tp = {i: 0 for i in range(6)}
        class_targets = {i: 0 for i in range(6)}

        for pred, target in zip(all_predictions, all_targets):
            target_boxes = target['boxes']
            target_labels = target['labels']

            # 统计总目标数
            for label in target_labels:
                total_targets += 1
                class_targets[label.item()] += 1

            # 应用置信度阈值
            score_mask = pred['scores'] > conf_thresh
            filtered_boxes = pred['boxes'][score_mask]
            filtered_labels = pred['labels'][score_mask]
            filtered_scores = pred['scores'][score_mask]

            total_predictions += len(filtered_boxes)

            # IoU匹配
            detected_targets = torch.zeros(len(target_boxes), dtype=torch.bool)

            for p_box, p_label, p_score in zip(filtered_boxes, filtered_labels, filtered_scores):
                matched = False
                for i, (t_box, t_label) in enumerate(zip(target_boxes, target_labels)):
                    if not detected_targets[i] and p_label == t_label:
                        iou = calculate_iou(p_box.numpy(), t_box.numpy())
                        if iou > iou_threshold:
                            detected_targets[i] = True
                            true_positives += 1
                            class_tp[t_label.item()] += 1
                            matched = True
                            break

                if not matched:
                    false_positives += 1

        # 计算指标
        precision = true_positives / total_predictions if total_predictions > 0 else 0.0
        recall = true_positives / total_targets if total_targets > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        false_positive_rate = false_positives / total_predictions if total_predictions > 0 else 0.0

        results[conf_thresh] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_positive_rate,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'total_targets': total_targets,
            'total_predictions': total_predictions,
            'class_recalls': {i: class_tp[i] / class_targets[i] if class_targets[i] > 0 else 0.0 for i in range(6)}
        }

        print(f"\nConf >= {conf_thresh:4.2f}:")
        print(f"  Predictions: {total_predictions:5d} | TP: {true_positives:4d} | FP: {false_positives:4d}")
        print(f"  Precision:   {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  FP Rate:     {false_positive_rate:.4f}")

    # 选择最佳阈值（基于F1分数）
    best_conf = max(confidence_thresholds, key=lambda x: results[x]['f1'])
    best_result = results[best_conf]

    print(f"\nBest threshold: {best_conf} (F1: {best_result['f1']:.4f})")

    return best_result['recall'], best_result['class_recalls']


def train_detection_head(infmae_model, train_dataloader, val_dataloader, device, epochs=1):
    """训练轻量级检测头"""

    # 使用InfMAE特征的检测器
    detector = LightweightDetectionHead(infmae_model, num_classes=6).to(device)

    # 只训练检测头，不训练InfMAE
    params = []
    for name, param in detector.named_parameters():
        if 'backbone.feature_extractor.infmae' not in name:  # 跳过InfMAE参数
            params.append(param)

    print(f"Training {len(params)} parameters (InfMAE frozen)")

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Training detection head for {epochs} epochs...")

    for epoch in range(epochs):
        detector.train()
        # 确保InfMAE保持eval模式
        detector.backbone.feature_extractor.infmae.eval()

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
                    valid_targets.append({
                        'boxes': target['boxes'].to(device).float(),
                        'labels': target['labels'].to(device).long()
                    })

            if not valid_frames: continue

            optimizer.zero_grad()

            try:
                with torch.amp.autocast('cuda'):
                    loss_dict = detector(valid_frames, valid_targets)
                    losses = sum(loss for loss in loss_dict.values())

                loss_value = losses.item()
                if not np.isfinite(loss_value):
                    print(f"\nWarning: NaN loss at batch {batch_idx}. Skipping.")
                    continue

                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss_value)

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue

            # 进度条
            percent = 100. * (batch_idx + 1) / total_batches
            bar_length = 40
            filled_len = int(bar_length * (batch_idx + 1) // total_batches)
            bar = '█' * filled_len + '-' * (bar_length - filled_len)
            print(f'\rProgress: |{bar}| {percent:.1f}% [{batch_idx + 1}/{total_batches}] - Loss: {loss_value:.4f}',
                  end='')
            if batch_idx + 1 == total_batches:
                print()

        lr_scheduler.step()
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch} summary - Average Loss: {avg_loss:.4f}")

        if val_dataloader is not None and (epoch + 1) % 2 == 0:
            recall, _ = comprehensive_evaluation(detector, val_dataloader, device, iou_threshold=0.3)
            print(f"Validation Recall after epoch {epoch}: {recall:.4f}")

    return detector


def run_validation_epoch(infmae_model, data_root, device, clip_length=16,
                         transform_train=None, transform_val=None,
                         det_epochs=1, data_ratio=1.0):
    print("=== Starting Validation Process (with InfMAE features) ===")
    original_training_state = infmae_model.training

    try:
        train_dataset = VideoDetectionDataset(
            data_root=data_root,
            scene_folders=['scene_1', 'scene_2', 'scene_3'],
            clip_length=clip_length,
            transform=transform_train,
            data_ratio=0.3
        )

        val_dataset = VideoDetectionDataset(
            data_root=data_root,
            scene_folders=['contest_1'],
            clip_length=clip_length,
            transform=transform_val,
            data_ratio=0.5
        )

        print(f"Detection training dataset: {len(train_dataset)} samples")
        print(f"Detection validation dataset: {len(val_dataset)} samples")

        if len(train_dataset) == 0:
            print("Warning: Training dataset is empty. Skipping validation.")
            return 0.0, {i: 0.0 for i in range(6)}

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=2,
            pin_memory=True, drop_last=True, collate_fn=detection_collate_fn
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=2, shuffle=False, num_workers=2,
            pin_memory=True, drop_last=False, collate_fn=detection_collate_fn
        )

        detector = train_detection_head(infmae_model, train_loader, val_loader, device, det_epochs)

        print("Running final comprehensive evaluation...")
        recall, class_recalls = comprehensive_evaluation(detector, val_loader, device, iou_threshold=0.3)

        return recall, class_recalls

    except Exception as e:
        print(f"Error during validation: {e}")
        traceback.print_exc()
        return 0.0, {i: 0.0 for i in range(6)}

    finally:
        infmae_model.train(original_training_state)
        print("=== Validation Process Completed ===")