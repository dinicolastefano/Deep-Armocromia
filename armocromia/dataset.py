from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ArmocromiaDataset(Dataset):
    """Dataset for Deep Armocromia images.

    Expects an annotations CSV with columns including `partition`, `class`, and
    image file paths (e.g. `path_rgb_original`). The CSV structure follows the
    README description.
    """

    def __init__(self, csv_path: str | Path, root_dir: str | Path,
                 partition: str = "train",
                 transform: Optional[Callable] = None) -> None:
        csv_path = Path(csv_path)
        root_dir = Path(root_dir)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["partition"] == partition]
        self.root = root_dir
        self.transform = transform

        # Map string labels to integer ids
        self.label_map = {label: i for i, label in
                          enumerate(sorted(self.data["class"].unique()))}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = self.root / row["path_rgb_original"]
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row["class"]]
        if self.transform:
            image = self.transform(image)
        return image, label
