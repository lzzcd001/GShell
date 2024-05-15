import torch
import numpy as np
from torch.utils.data import Dataset

class GShellDataset(Dataset):
    def __init__(self, filepath_metafile, extension='pt'):
        super().__init__()
        with open(filepath_metafile, 'r') as f:
            self.filepath_list = [fpath.rstrip() for fpath in f]

        self.extension = extension
        assert self.extension in ['pt', 'npy']
    
    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        with torch.no_grad():
            if self.extension == 'pt':
                datum = torch.load(self.filepath_list[idx], map_location='cpu')
            else:
                datum = torch.tensor(np.load(self.filepath_list[idx]))
        return datum
