# util/video_pretrain_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


def get_image_paths(directory):
    """Helper function to get sorted list of image paths from a directory."""
    image_extensions = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    try:
        paths = [
            os.path.join(directory, f) for f in sorted(
                os.listdir(directory),
                key=lambda x: int(os.path.splitext(x)[0])
            ) if f.lower().endswith(image_extensions)
        ]
    except ValueError:
        paths = [
            os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(image_extensions)
        ]
    return paths


class VideoMAEPretrainDataset(Dataset):
    """Dataset for VideoMAE-style pre-training."""

    def __init__(self, data_root, scene_folders, clip_length=16, transform=None, data_ratio=1.0):
        """
        Args:
            data_root (str): The root directory of the dataset.
            scene_folders (list): List of scene folder names.
            clip_length (int): Number of frames in each clip.
            transform (callable, optional): A function/transform to be applied on each frame.
            data_ratio (float): Ratio of data to use (0.0-1.0).
        """
        self.clip_length = clip_length
        self.transform = transform
        self.samples = []

        print(f"Initializing dataset with {data_ratio * 100}% data ratio...")

        # Collect all sequences first
        all_sequences = []
        for scene_name in scene_folders:
            scene_path = os.path.join(data_root, scene_name)
            images_root = os.path.join(scene_path, 'images')
            if not os.path.isdir(images_root):
                print(f"Warning: 'images' directory not found in {scene_path}")
                continue

            for seq_name in sorted(os.listdir(images_root)):
                seq_path = os.path.join(images_root, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                frame_paths = get_image_paths(seq_path)
                if len(frame_paths) >= self.clip_length:
                    all_sequences.append(frame_paths)

        # Apply data ratio
        if data_ratio < 1.0:
            num_sequences = max(1, int(len(all_sequences) * data_ratio))
            all_sequences = random.sample(all_sequences, num_sequences)

        # Create clips from selected sequences
        for frame_paths in all_sequences:
            num_frames = len(frame_paths)
            for i in range(num_frames - self.clip_length + 1):
                self.samples.append((frame_paths, i))

        if not self.samples:
            raise RuntimeError(f"Found 0 video clips in {data_root}. Please check dataset path and structure.")

        print(f"Dataset initialized. Found {len(self.samples)} possible video clips.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, start_index = self.samples[idx]

        clip_frames = []
        for i in range(self.clip_length):
            frame_path = frame_paths[start_index + i]
            try:
                with Image.open(frame_path).convert('RGB') as img:
                    if self.transform:
                        img = self.transform(img)
                    clip_frames.append(img)
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                return torch.zeros(self.clip_length, 3, 224, 224), torch.tensor(0)

        return torch.stack(clip_frames, dim=0), torch.tensor(0)