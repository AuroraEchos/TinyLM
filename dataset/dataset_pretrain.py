<<<<<<< HEAD
import torch
import numpy as np
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(
        self,
        bin_path: str,
        max_length: int = 512,
        split: str = "train",
        val_ratio: float = 0.01,   # 1% 做验证
        seed: int = 42,
    ):
        assert split in ["train", "val"]

        self.data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.max_length = max_length
        self.split = split

        total_tokens = len(self.data)

        # ===== 1️⃣ token 级别划分 =====
        val_tokens = int(total_tokens * val_ratio)

        if split == "train":
            self.start = 0
            self.end = total_tokens - val_tokens
        else:
            self.start = total_tokens - val_tokens
            self.end = total_tokens

        self.split_length = self.end - self.start

        # ===== 2️⃣ 可生成多少 sample =====
        self.num_samples = (self.split_length - 1) // max_length

        # ===== 3️⃣ 训练才随机 offset =====
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        if self.split == "train":
            offset = self.rng.randint(0, self.max_length)
        else:
            offset = 0  # 验证集 deterministic

        start = self.start + idx * self.max_length + offset
        end = start + self.max_length + 1

        chunk = self.data[start:end].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

        return input_ids, labels


if __name__ == "__main__":
    train_ds = PretrainDataset(
        "data/pretrain_data.bin",
        split="train",
        val_ratio=0.01,
    )

    val_ds = PretrainDataset(
        "data/pretrain_data.bin",
        split="val",
        val_ratio=0.01,
    )

    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))

    x, y = val_ds[0]
    print("Val sample shape:", x.shape)
=======
from torch.utils.data import Dataset
import torch
import numpy as np
from datasets import load_dataset

class PretrainDataset(Dataset):
    def __init__(self, bin_path, max_length=512):
        super().__init__()
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.max_length = max_length

    def __len__(self):
        return len(self.data) // self.max_length

    def __getitem__(self, index):
        # 截取一段长度为 max_length 的数据
        start = index * self.max_length
        end = start + self.max_length
        
        chunk = self.data[start:end].astype(np.int64)
        
        input_ids = torch.from_numpy(chunk)
        # 预训练中 labels 通常就是 input_ids 的克隆
        # 偏移（Shift）操作通常由模型的 Forward 或 Loss 函数处理
        labels = input_ids.clone() 
        
        return input_ids, labels
    
if __name__ == "__main__":
    dataset = PretrainDataset('data/pretrain_hq.bin', max_length=512)
    print(f"Dataset size: {len(dataset)} samples")
    input_ids, labels = dataset[0]
    print(f"Sample input_ids: {input_ids}")
    print(f"Sample labels: {labels}")
>>>>>>> da8a0fd5f7950b005684b0a094219ef543e49ae8
