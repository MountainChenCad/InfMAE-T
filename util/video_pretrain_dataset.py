# util/video_pretrain_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


def get_image_paths(directory):
    """Helper function to get sorted list of image paths from a directory."""
    image_extensions = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    # Sort files numerically based on their names (e.g., '1.bmp', '2.bmp', '10.bmp')
    try:
        paths = [
            os.path.join(directory, f) for f in sorted(
                os.listdir(directory),
                key=lambda x: int(os.path.splitext(x)[0])
            ) if f.lower().endswith(image_extensions)
        ]
    except ValueError:
        # Fallback to alphabetical sorting if filenames are not purely numeric
        paths = [
            os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(image_extensions)
        ]
    return paths


class VideoMAEPretrainDataset(Dataset):
    """
    Dataset for VideoMAE-style pre-training.
    Each sample is a clip of T consecutive frames from the specified dataset structure.
    Dataset structure is expected to be: data_path/scene_X/images/dataY/*.bmp
    """

    def __init__(self, data_root, scene_folders, clip_length=16, transform=None):
        """
        Args:
            data_root (str): The root directory of the dataset (e.g., './Dataset').
            scene_folders (list): List of scene folder names (e.g., ['scene_1', 'scene_2']).
            clip_length (int): Number of frames in each clip.
            transform (callable, optional): A function/transform to be applied on each frame.
        """
        self.clip_length = clip_length
        self.transform = transform
        self.samples = []

        print("Initializing dataset... this may take a moment.")
        # 1. Iterate through each provided scene path
        for scene_name in scene_folders:
            scene_path = os.path.join(data_root, scene_name)
            images_root = os.path.join(scene_path, 'images')
            if not os.path.isdir(images_root):
                print(f"Warning: 'images' directory not found in {scene_path}")
                continue

            # 2. Iterate through each data sequence (data1, data2, ...)
            for seq_name in sorted(os.listdir(images_root)):
                seq_path = os.path.join(images_root, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                frame_paths = get_image_paths(seq_path)
                num_frames = len(frame_paths)

                # 3. If a sequence is long enough, create all possible start indices for clips
                if num_frames >= self.clip_length:
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
        # Load the T frames for the clip
        for i in range(self.clip_length):
            frame_path = frame_paths[start_index + i]
            try:
                with Image.open(frame_path).convert('RGB') as img:
                    if self.transform:
                        img = self.transform(img)
                    clip_frames.append(img)
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                # Return a dummy tensor if an image is corrupted, to avoid crashing the training loop
                return torch.zeros(self.clip_length, 3, 224, 224)

        # Stack frames to form a (T, C, H, W) tensor
        return torch.stack(clip_frames, dim=0), torch.tensor(0)  # Return a dummy label