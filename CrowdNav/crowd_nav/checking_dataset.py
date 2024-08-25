import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def append(self, item):
        self.data.append(item)

# Initialize your dataset and data loader
memory = CustomDataset()
memory.append((torch.tensor([1.0]), torch.tensor([1.0])))
data_loader = DataLoader(memory, batch_size=2, shuffle=True)
# Add initial data
memory.append((torch.tensor([2.0]), torch.tensor([2.0])))

import numpy as np

# Now let's iterate over the data loader
for inputs, values in data_loader:
    print(f'input :\n{np.array(inputs)}')
    print(f'values :\n {np.array(values)}')


i = 0
print(f'Done with {i} loop')
i+=1
# Add new data to the dataset
memory.append((torch.tensor([3.0]), torch.tensor([3.0])))

# Iterate again to see if the new data is included
for inputs, values in data_loader:
    print(f'input :\n{np.array(inputs)}')
    print(f'values :\n {np.array(values)}')
print(f'Done with {i} loop')
i+=1
# Using next(iter(data_loader)) to get a single batch
for j in range(2):
    inputs, values = next(iter(data_loader))
    print(f'input :\n{np.array(inputs)}')
    print(f'values :\n {np.array(values)}')
    print(f'Done with {i}.{j} loop')
i+=1
