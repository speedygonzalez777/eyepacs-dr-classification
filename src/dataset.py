from pathlib import Path 

import cv2
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset

from src.preprocessing import (
    preprocess_crop_resize,
    preprocess_crop_ben_resize,
    preprocess_crop_clahe_resize,
    preprocess_crop_greenben_resize,
)


PREPROCESSING_FUNCTIONS = {
    'crop_resize': preprocess_crop_resize,
    'crop_ben_resize': preprocess_crop_ben_resize,
    'crop_clahe_resize': preprocess_crop_clahe_resize,
    'crop_greenben_resize': preprocess_crop_greenben_resize,
}


class EyePACSDataset(Dataset):
    def __init__(
        self,
        labels_df: pd.DataFrame,
        images_dir: str | Path,
        preprocessing: str = "crop_resize",
        image_size: int = 512,
        use_augmentations: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.labels_df = labels_df.reset_index(drop=True).copy()
        
        if "binary_label" not in self.labels_df.columns:
            self.labels_df["binary_label"] = (self.labels_df["level"] > 0).astype(int)

        self.images_dir = Path(images_dir)
        self.preprocessing = preprocessing
        self.image_size = image_size
        self.use_augmentations = use_augmentations
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        if preprocessing not in PREPROCESSING_FUNCTIONS:
            available = ", ".join(PREPROCESSING_FUNCTIONS.keys())
            raise ValueError(
                f"Unknown preprocessing: {preprocessing}. "
                f"Available options: {available}"
            )

        self.preprocessing_fn = PREPROCESSING_FUNCTIONS[preprocessing]

    def __len__(self) -> int:
        return len(self.labels_df)

    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))

        angle = np.random.uniform(-15.0, 15.0)
        center = (self.image_size / 2.0, self.image_size / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image,
            rotation_matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if np.random.rand() < 0.3:
            zoom = np.random.uniform(1.0, 1.05)
            zoomed_size = int(round(self.image_size * zoom))
            zoomed = cv2.resize(
                image,
                (zoomed_size, zoomed_size),
                interpolation=cv2.INTER_LINEAR,
            )
            max_offset = zoomed_size - self.image_size
            top = np.random.randint(0, max_offset + 1)
            left = np.random.randint(0, max_offset + 1)
            image = zoomed[
                top:top + self.image_size,
                left:left + self.image_size,
            ]

        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10.0, 10.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255)

        return image.astype(np.uint8)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels_df.iloc[idx]

        image_name = row['image']
        image_path = self.images_dir / f'{image_name}.jpeg'

        image = self.preprocessing_fn(image_path, size=self.image_size)
        if self.use_augmentations:
            image = self._apply_augmentations(image)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        image = (image - self.mean) / self.std 

        label = torch.tensor(row['binary_label'], dtype=torch.float32)

        return image, label 
        
