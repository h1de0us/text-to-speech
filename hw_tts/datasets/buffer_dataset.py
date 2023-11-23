from torch.utils.data import Dataset

from hw_tts.utils import get_data_to_buffer
from typing import List

class BufferDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 mel_ground_truth: str,
                 alignment_path: str,
                 text_cleaners: List[str],
                 batch_expand_size: int = 1,
                ):
        self.buffer = get_data_to_buffer(
            data_path,
            mel_ground_truth,
            alignment_path,
            text_cleaners,
            batch_expand_size
        )
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]