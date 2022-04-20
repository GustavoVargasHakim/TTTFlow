import torch
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, n):
        super(Data, self).__init__()
        self.n = n
    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.rand(3, 4, 4)

        return {'image': x}

dataset = Data(10)
dataloader = DataLoader(dataset, batch_size=2)
for i, data in enumerate(dataloader, 0):
    im = data['image']
    x = torch.flatten(im, start_dim=1, end_dim=3)
    break