import numpy as np
import pandas as pd
import torch as torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data import ChallengeDataset
from model import ResNet
from trainer import Trainer

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data = pd.read_csv('data.csv', sep=';')
data_train, data_test = train_test_split(data, test_size=0.10)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

train_dl = torch.utils.data.DataLoader(ChallengeDataset(data_train, 'train'), batch_size=25)
val_dl = torch.utils.data.DataLoader(ChallengeDataset(data_test, 'val'), batch_size=25)
# create an instance of our ResNet model
# TODO

resnet = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO

loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(resnet.parameters())
trainer = Trainer(resnet, loss, optim=optimizer, train_dl=train_dl, val_test_dl=val_dl, early_stopping_patience=10)

# go, go, go... call fit on trainer
res = trainer.fit(1000)  # TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
