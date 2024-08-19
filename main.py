import torch
import numpy as np
import lightning as L
import wandb
from torch.utils.data import DataLoader
from models import CanonizationNetwork, SymmetrizationNetwork, IntrinsicNetwork, StandardNetwork
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=10000, n=10, d=5, std=0.8, augment=False):
        self.n_samples = n_samples
        self.n = n
        self.d = d
        self.data = torch.concatenate(
            [torch.randn(n_samples // 2, n, d), torch.randn((n_samples - (n_samples // 2)), n, d) * std]).view(
            n_samples,
            n, d)
        self.augment = augment
        self.labels = np.ones(n_samples)
        self.labels[self.n_samples // 2:] = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.augment:
            perm = torch.randperm(self.n)
            return self.data[idx, perm, :], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    train_val_split = 0.8
    d = 5
    batch_size = 256
    channels = (d, 64, 128, 1)

    for n_samples in [100, 1000, 10000]:
        for n in [10, 100]:
            train_dataset = GaussianDataset(int(n_samples * train_val_split), n=n, d=d)
            val_dataset = GaussianDataset(int(n_samples * (1 - train_val_split)), n=n, d=d)
            train_dataset_aug = GaussianDataset(int(n_samples * train_val_split), n=n, d=d, augment=True)
            val_dataset_aug = GaussianDataset(int(n_samples * (1 - train_val_split)), n=n, d=d, augment=True)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            train_aug_loader = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True, num_workers=8)
            val_aug_loader = DataLoader(val_dataset_aug, batch_size=batch_size, shuffle=False, num_workers=8)

            models = [StandardNetwork(n, 1, channels),
                      CanonizationNetwork(n, channels),
                      SymmetrizationNetwork(n, 1, channels, 100),
                      SymmetrizationNetwork(n, 1, channels, 1000),
                      IntrinsicNetwork(n, 1, channels)]
            for model in models:
                model_name = type(model).__name__
                run_name = f'{model_name}_n_samples_{n_samples}_n_{n}_d_{d}'
                # early_stopping = EarlyStopping('val_acc', patience=15)
                wandb_logger = WandbLogger(log_model='all', name=run_name)
                wandb_logger.log_hyperparams({
                    'n_samples': n_samples,
                    'n': n,
                    'd': d,
                    'model': model_name
                })
                trainer = L.Trainer(max_epochs=250, logger=wandb_logger,
                                    check_val_every_n_epoch=10)  # callbacks=[early_stopping],
                tloader = train_aug_loader if type(model) == StandardNetwork else train_loader
                vloader = val_aug_loader if type(model) == StandardNetwork else val_loader
                trainer.fit(model=model, train_dataloaders=tloader, val_dataloaders=vloader)
                # finish the run
                wandb.finish()
