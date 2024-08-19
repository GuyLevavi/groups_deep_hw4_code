import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from itertools import permutations
from torch.utils.data import DataLoader
import lightning as L


def inv_perm(a):
    return torch.argsort(a)


class StandardNetwork(nn.Module):
    # used with augmentation
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        assert output_dim == input_dim or output_dim == 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x - (B, n, d) where B is batch size, d is input dimension, n is number of points
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)


class CanonizationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x - (B, n, d) where B is batch size, d is input dimension, n is number of points
        # canonization is simply sorting the input
        indices = torch.argsort(x, dim=1)
        x = torch.gather(x, 1, indices)
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)


class SymmetrizationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_sampled_permutations=None):
        super().__init__()
        assert output_dim == input_dim or output_dim == 1  # equivariant or invariant
        self.type = 'equivariant' if output_dim == input_dim else 'invariant'
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if n_sampled_permutations is not None:
            self.n_sampled_permutations = n_sampled_permutations
            self.perms = None
            SymmetrizationNetwork.__name__ += 'Sampled'
        else:
            self.perms = [torch.LongTensor(perm) for perm in list(permutations(range(input_dim)))]
            self.n_sampled_permutations = None



    def forward_single(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)

    def forward(self, x):
        # x - (B, n, d) where B is batch size, d is input dimension, n is number of points
        # symmetrization is simply averaging the input across all permutations

        if self.n_sampled_permutations is not None:  # sample permutations
            perms = [torch.randperm(x.shape[1]) for _ in range(self.n_sampled_permutations)]
        else:  # all permutations
            perms = self.perms

        if self.type == 'invariant':
            x = torch.stack([self.forward_single(x[:, perm, :]) for perm in perms], dim=0)
        else:  # equivariant
            x = torch.stack([self.forward_single(x[:, perm, :])[:, inv_perm(perm), :] for perm in perms], dim=0)
        x = x.mean(dim=0)
        return x


class EquivariantLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        assert output_dim == input_dim or output_dim == 1
        self.type = 'equivariant' if output_dim == input_dim else 'invariant'
        self.alpha = nn.Parameter(torch.Tensor(1)) if self.type == 'equivariant' else None
        self.beta = nn.Parameter(torch.Tensor(1))
        self.bias = nn.Parameter(torch.Tensor(output_dim))  # in case of equivariant will be broadcasted
        self.reset_parameters()

    def reset_parameters(self):
        if self.type == 'equivariant':
            nn.init.constant_(self.alpha, 1)
        nn.init.constant_(self.beta, 1)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # alpha * x + beta * 1 * 1^T * x + bias * 1
        # where 1 is a vector of ones
        mean = x.mean(dim=1, keepdim=True)
        if self.type == 'equivariant':
            x = self.alpha * x + self.beta * mean + self.bias[None, :, None]
        else:
            x = self.beta * mean + self.bias[None, :, None]
        return x


class InternalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = EquivariantLinearLayer(input_dim, input_dim)
        self.fc2 = EquivariantLinearLayer(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def is_equivariant(network, input_dim, feature_dim, n_trials):
    # checks empirically if the network is equivariant or invariant
    x = torch.randn(n_trials, input_dim, feature_dim)
    perms = [torch.randperm(input_dim) for _ in range(n_trials)]
    # for each trial apply a random permutation
    x_permed = torch.stack([x[i, perm, :] for i, perm in enumerate(perms)], dim=0)
    y_input_permed = network(x_permed)
    type = 'equivariant' if y_input_permed.shape[1] == input_dim else 'invariant'
    y = network(x)
    if type == 'equivariant':
        y_permed = torch.stack([y[i, perm, :] for i, perm in enumerate(perms)], dim=0)
    else:
        y_permed = y
    res = torch.allclose(y_input_permed, y_permed, atol=1e-3)
    print('{} is {} : {}'.format(network.__class__.__name__, type, res))
    # print('perms:\n', perms[0])
    # print('input:\n', x[0, :, :])
    # print('input_permed:\n', x_permed[0, :, :])
    # print('out(input):\n', y[0, :, :])
    # print('out(input_permed):\n', y_input_permed[0, :, :])



if __name__ == '__main__':
    torch.manual_seed(1)
    n = 5
    d = 3
    n_trials = 100
    is_equivariant(InternalNetwork(n, n), n, d, n_trials)
    is_equivariant(InternalNetwork(n, 1), n, d, n_trials)
    is_equivariant(SymmetrizationNetwork(n, 10, n), n, d, n_trials)
    is_equivariant(SymmetrizationNetwork(n, 10, 1), n, d, n_trials)
    is_equivariant(SymmetrizationNetwork(n, 10, n, 100), n, d, n_trials)
    is_equivariant(SymmetrizationNetwork(n, 10, 1, 100), n, d, n_trials)
    is_equivariant(CanonizationNetwork(n, n), n, d, n_trials)
    is_equivariant(StandardNetwork(n, 2*n, n), n, d, n_trials)