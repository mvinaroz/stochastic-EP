import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import argparse
from torch.distributions import Gamma
import pickle
import numpy.random as rn
from sklearn.metrics import roc_auc_score

from dp_sgd import dp_sgd_backward
from backpack import extend

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(Feedforward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        # self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size2)
        # self.bn2 = nn.BatchNorm1d(self.hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        # hidden = self.bn1(hidden)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        # output = self.bn2(output)
        output = self.fc3(self.relu(output))
        output = self.sigmoid(output)
        return output

def main():

    ar = parse()
    rn.seed(ar.seed)

    """ load data """
    filename = 'adult.p'
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # unpack data
    y_tot, x_tot = data
    N_tot, input_dim = x_tot.shape

    use_cuda = not ar.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # train and test data
    N = np.int(N_tot * 0.9)
    print('n_data:', N)
    rand_perm_nums = np.random.permutation(N_tot)
    X = x_tot[rand_perm_nums[0:N], :]
    y = y_tot[rand_perm_nums[0:N]]
    Xtst = x_tot[rand_perm_nums[N:], :]
    ytst = y_tot[rand_perm_nums[N:]]

#############################################################
    """ train a classifier """
#############################################################
    classifier = Feedforward(input_dim, 100, 20)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=ar.lr)

    classifier.train()
    if ar.dp_sigma > 0.:
        extend(classifier)
    # how_many_epochs = 10
    # mini_batch_size = 100
    how_many_iter = np.int(N / ar.clf_batch_size)

    for epoch in range(ar.clf_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):
            # get the inputs
            inputs = X[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size, :]
            labels = y[i * ar.clf_batch_size:(i + 1) * ar.clf_batch_size]

            optimizer.zero_grad()
            y_pred = classifier(torch.Tensor(inputs))
            loss = criterion(y_pred.squeeze(), torch.FloatTensor(labels))

            if ar.dp_sigma > 0.:
                global_norms, global_clips = dp_sgd_backward(classifier.parameters(), loss, device, ar.dp_clip, ar.dp_sigma)
                # print(f'max_norm:{torch.max(global_norms).item()}, mean_norm:{torch.mean(global_norms).item()}')
                # print(f'mean_clip:{torch.mean(global_clips).item()}')
            else:
                loss.backward()
            optimizer.step()
            running_loss += loss.item()

        y_pred = classifier(torch.Tensor(Xtst))
        ROC = roc_auc_score(ytst, y_pred.detach().numpy())
        print('Epoch {}: ROC : {}'.format(epoch, ROC))

    print('Finished Classifier Training')




def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--clf-epochs', type=int, default=20)
    parser.add_argument('--clf-batch-size', type=int, default=1000)

    parser.add_argument('--save-model', action='store_true', default=False)

    parser.add_argument('--dp-clip', type=float, default=0.01)

    parser.add_argument('--dp-sigma', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()