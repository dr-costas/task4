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
        return self.data.dataset.num_examples

    @staticmethod
    def collate(batch):
        feats, targets1, targets2 = zip(*batch)
        # Pad features
        max_len_x = int(np.max([sample.size(1) for sample in feats]))
        X = torch.FloatTensor(len(feats), max_len_x, feats[0].size(2)).zero_()
        max_len_y1 = int(np.max([sample.size(1) for sample in targets1]))
        Y1 = torch.ByteTensor(len(targets1), max_len_y1, targets1[0].size(2)).zero_()
        max_len_y2 = int(np.max([sample.size(1) for sample in targets2]))
        Y2 = torch.ByteTensor(len(targets2), max_len_y2, targets2[0].size(2)).zero_()
        for k, (x, y1, y2) in enumerate(zip(feats, targets1, targets2)):
            X[k, :x.size(1), :] = x
            Y1[k, :y1.size(1), :] = y1
            Y2[k, :y2.size(2), :] = y2
        return X, Y1, Y2


if __name__ == '__main__':
    import pickle
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    dataset = ChallengeDataset(dataset_name='attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5',
            that_set='train', scaler=scaler)
    x = dataset[0]

    # Test DataLoader with collate function
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=ChallengeDataset.collate)
    x, y1, y2 = next(loader.__iter__())

