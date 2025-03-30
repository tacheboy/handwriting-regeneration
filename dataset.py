import torch
from torch.utils.data import Dataset

class HandwritingDataset(Dataset):
    def __init__(self, stroke_data):
        """
        stroke_data: numpy array of shape (num_samples, seq_length, features)
        Creates input/target pairs for next-step prediction.
        """
        self.data = stroke_data
        self.num_samples = stroke_data.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Create input (all timesteps except last) and target (all timesteps shifted by one)
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        return torch.tensor(input_seq), torch.tensor(target_seq)
