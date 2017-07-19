import torch
from torch.utils.data import Dataset
from fuel.datasets import H5PYDataset
from fuel.schemes import *
from fuel.streams import DataStream
import numpy as np

class ChallengeDataset(Dataset):

    def __init__(self, dataset_name, that_set, scaler):
        super(ChallengeDataset, self).__init__()
        raw_data = H5PYDataset(dataset_name, which_sets=(that_set, ), load_in_memory=False)
        scheme = SequentialExampleScheme(raw_data.num_examples)
        self.data = DataStream(dataset=raw_data, iteration_scheme=scheme)
        self.scaler = scaler

    def __getitem__(self, idx):
        x, _, _, y1, y2 = self.data.get_data(idx)
        x_norm = self.scaler.transform(x.squeeze())
        return torch.FloatTensor(x_norm[np.newaxis]), torch.ByteTensor(y1), torch.ByteTensor(y2)

    def __len__(self):
        return self.data.num_examples

if __name__ == '__main__':
    import pickle
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    dataset = ChallengeDataset(dataset_name='attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5',
            that_set='train', scaler=scaler)
    x = dataset[0]

