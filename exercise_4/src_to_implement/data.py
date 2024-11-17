import torch
import torchvision as tv
from skimage.color import gray2rgb
from skimage.io import imread
from torch.utils.data import Dataset

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # imread(self.data.iloc[index, 0])
        img = self._transform(gray2rgb(imread(self.data.iloc[index, 0]))).float()
        labels = torch.tensor((self.data.iloc[index, 1], self.data.iloc[index, 2])).float()
        return img, labels
