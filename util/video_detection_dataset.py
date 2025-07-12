# util/video_detection_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np


def get_image_paths(directory):
    """获取目录下已排序的图像路径列表。"""
    if not os.path.exists(directory):
        return []

    image_extensions = ('.bmp', '.png', '.jpg', '.jpeg')
    paths = []

    try:
        # 尝试按文件名中的数字排序
        files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

        # 分离带数字和不带数字的文件
        numeric_files = []
        other_files = []
        for f in files:
            try:
                numeric_files.append((int(os.path.splitext(f)[0]), f))
            except ValueError:
                other_files.append(f)

        # 对数字文件排序，然后添加其他文件
        numeric_files.sort()
        sorted_files = [f for _, f in numeric_files] + sorted(other_files)

        return [os.path.join(directory, f) for f in sorted_files]
    except Exception as e:
        print(f"Error getting image paths from {directory}: {e}")
        return []


def parse_txt_label(txt_path, img_width, img_height):
    """解析YOLO格式的TXT标签文件。"""
    if not os.path.exists(txt_path):
        return []

    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = map(float, parts)

                # 从归一化坐标转换为绝对像素坐标
                abs_x_center = x_center * img_width
                abs_y_center = y_center * img_height
                abs_width = width * img_width
                abs_height = height * img_height

                # 从中心点+宽高转换为(xmin, ymin, xmax, ymax)
                xmin = int(abs_x_center - (abs_width / 2))
                ymin = int(abs_y_center - (abs_height / 2))
                xmax = int(abs_x_center + (abs_width / 2))
                ymax = int(abs_y_center + (abs_height / 2))

                # 确保边界框有效
                if xmax > xmin and ymax > ymin:
                    boxes.append({
                        'class_id': int(class_id),
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
            except Exception as e:
                print(f"Error parsing line in {txt_path}: '{line.strip()}' -> {e}")
                continue
    return boxes


def detection_collate_fn(batch):
    """检测数据集的自定义collate函数。"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    clips = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]

    if not clips:
        return None

    return torch.stack(clips, dim=0), targets_list


class VideoDetectionDataset(Dataset):
    """用于视频检测验证的数据集，现在支持TXT标签。"""

    def __init__(self, data_root, scene_folders, clip_length=16, transform=None, data_ratio=1.0):
        self.clip_length = clip_length
        self.transform = transform
        self.samples = []

        print(f"Loading detection dataset with TXT labels (Ratio: {data_ratio * 100:.1f}%)")

        if not os.path.isdir(data_root):
            print(f"ERROR: Data root directory does not exist: {data_root}")
            return

        for scene_name in scene_folders:
            scene_path = os.path.join(data_root, scene_name)
            if not os.path.isdir(scene_path):
                print(f"Warning: Scene directory not found: {scene_path}")
                continue

            images_root = os.path.join(scene_path, 'images')
            labels_root = os.path.join(scene_path, 'labels')

            if not os.path.isdir(images_root) or not os.path.isdir(labels_root):
                continue

            data_folders = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])

            for data_folder in data_folders:
                img_folder = os.path.join(images_root, data_folder)
                label_folder = os.path.join(labels_root, data_folder)

                if not os.path.isdir(label_folder):
                    continue

                frame_paths = get_image_paths(img_folder)
                if len(frame_paths) >= self.clip_length:
                    # 只添加那些至少有一个对应标签文件的clip
                    step = max(1, self.clip_length // 2)
                    for i in range(0, len(frame_paths) - self.clip_length + 1, step):
                        # 检查这个clip中是否有任何一个标签文件存在
                        has_any_label = False
                        for j in range(i, i + self.clip_length):
                            frame_name = os.path.basename(frame_paths[j])
                            label_name = os.path.splitext(frame_name)[0] + '.txt'
                            if os.path.exists(os.path.join(label_folder, label_name)):
                                has_any_label = True
                                break

                        if has_any_label:
                            self.samples.append((frame_paths, label_folder, i))

        # 应用数据比例
        if data_ratio < 1.0 and len(self.samples) > 0:
            num_samples = int(len(self.samples) * data_ratio)
            self.samples = random.sample(self.samples, num_samples)

        print(f"Detection dataset loaded: {len(self.samples)} clips")
        if len(self.samples) == 0:
            print("WARNING: No valid clips found! Check dataset structure and label format (.txt).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            frame_paths, label_folder, start_idx = self.samples[idx]

            frames = []
            targets = []

            first_img = Image.open(frame_paths[start_idx]).convert('RGB')
            img_width, img_height = first_img.size

            for i in range(self.clip_length):
                frame_path = frame_paths[start_idx + i]

                # 加载图像
                try:
                    # 避免重复打开第一张图
                    if i == 0:
                        img = first_img
                    else:
                        img = Image.open(frame_path).convert('RGB')

                    if self.transform:
                        img_tensor = self.transform(img)
                    else:
                        img_tensor = torch.from_numpy(np.array(img))
                    frames.append(img_tensor)
                except Exception as e:
                    print(f"Error loading image {frame_path}: {e}")
                    return None

                # 加载标签
                label_name = os.path.splitext(os.path.basename(frame_path))[0] + '.txt'
                label_path = os.path.join(label_folder, label_name)

                boxes_data = parse_txt_label(label_path, img_width, img_height)

                target = {
                    'boxes': torch.tensor([b['bbox'] for b in boxes_data],
                                          dtype=torch.float32) if boxes_data else torch.zeros((0, 4),
                                                                                              dtype=torch.float32),
                    'labels': torch.tensor([b['class_id'] for b in boxes_data],
                                           dtype=torch.int64) if boxes_data else torch.zeros((0,), dtype=torch.int64)
                }
                targets.append(target)

            return torch.stack(frames), targets

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            return None