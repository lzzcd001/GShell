import torch
from torch.utils.data import Dataset

class GShellAugDataset(Dataset):
    def __init__(self, FLAGS, extension='pt'):
        super().__init__()
        with open(FLAGS.data.grid_metafile, 'r') as f:
            self.filepath_list = [fpath.rstrip() for fpath in f]
        with open(FLAGS.data.occgrid_metafile, 'r') as f:
            self.occ_filepath_list = [fpath.rstrip() for fpath in f]

        self.extension = extension
        self.num_channels = FLAGS.data.num_channels
        print('num_channels: ', self.num_channels)
        assert self.extension in ['pt', 'npy']
    
    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        with torch.no_grad():
            grid = torch.load(self.filepath_list[idx], map_location='cpu')
            try:
                occ_grid = torch.load(self.occ_filepath_list[idx], map_location='cpu')
            except:
                print(self.occ_filepath_list[idx])
                raise
        return (grid[:self.num_channels], occ_grid)
    
    @staticmethod
    def collate(data):
        return {
            'grid': torch.stack([x[0] for x in data]),
            'occgrid': torch.stack([x[1] for x in data]),
        }
