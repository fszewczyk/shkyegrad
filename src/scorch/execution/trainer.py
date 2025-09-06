from typing import Union
import torch
import numpy as np

class Runner:
    def __init__(
        self,
        model,
        training_loader,
        optimizer,
        loss,
        epochs: int,
        validation_loader = None,
        loggers = (),
        lr_scheduler = None,
    ):
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.loss_fn = loss
        self.loggers = loggers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

    def train(self):
        return self.run(train=True, validate=True)
    
    def validate(self):
        return self.run(train=False, validate=True)

    def run(self, train=True, validate=True):
        iters = self.epochs if train else 1
        self.model.to(self.device)
        for epoch in range(iters):
            if train:
                self.model.train()
                for batch_index, (x, y) in enumerate(self.training_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad()

                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                    loss.backward()

                    self.optimizer.step()

                    inference_data = {
                        'input': x,
                        'target': y,
                        'prediction': pred,
                        'loss': loss.detach(),
                        'epoch': epoch,
                        'total_epochs': iters,
                        'last_epoch': epoch == iters - 1,
                        'batch_index': batch_index,
                        'global_batch_index': epoch * len(self.training_loader) + batch_index,
                        'train': True,
                        'model': self.model,
                        'optimizer': self.optimizer,
                        'lr_scheduler': self.lr_scheduler
                    }
                    for logger in self.loggers:
                        logger(inference_data)

            if validate:
                if self.validation_loader:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_index, (x, y) in enumerate(self.validation_loader):
                            x = x.to(self.device)
                            y = y.to(self.device)

                            pred = self.model(x)
                            loss = self.loss_fn(pred, y)
                            
                            inference_data = {
                                'input': x,
                                'target': y,
                                'prediction': pred,
                                'loss': loss.detach(),
                                'epoch': epoch,
                                'total_epochs': iters,
                                'last_epoch': epoch == iters - 1,
                                'batch_index': batch_index,
                                'global_batch_index': epoch * len(self.validation_loader) + batch_index,
                                'train': False,
                                'model': self.model,
                                'optimizer': self.optimizer,
                                'lr_scheduler': self.lr_scheduler
                            }
                            for logger in self.loggers:
                                logger(inference_data)

            if self.lr_scheduler:
                self.lr_scheduler.step()


