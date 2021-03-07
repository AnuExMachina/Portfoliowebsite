import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class TitanicNN(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(8, 16)
        self.dense2 = nn.Linear(16, 12)
        self.dense3 = nn.Linear(12, 8)
        self.dense4 = nn.Linear(8, 4)
        self.dense5 = nn.Linear(4, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
    def forward(self, x):
        x = F.gelu(self.dense1(x))
        x = self.dropout1(x)
        x = F.gelu(self.dense2(x))
        x = self.dropout2(x)
        x = F.gelu(self.dense3(x))
        x = self.dropout3(x)
        x = F.gelu(self.dense4(x))
        x = self.dropout4(x)
        x = F.sigmoid(self.dense5(x))
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())  