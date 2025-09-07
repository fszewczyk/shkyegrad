from typing import Union
import torch
import numpy as np
from ..utils.profiler import Profiler

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
                    with Profiler() as profiler_data_move:
                        x = x.to(self.device)
                        y = y.to(self.device)

                    self.optimizer.zero_grad()

                    with Profiler() as profiler_inference:
                        pred = self.model(x)

                    with Profiler() as profiler_loss:
                        loss = self.loss_fn(pred, y)

                    with Profiler() as profiler_backward:
                        loss.backward()

                    with Profiler() as profiler_optimizer:
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
                        'lr_scheduler': self.lr_scheduler,
                        'profiles': {
                            'batch_load': self.training_loader.last_waiting_time if hasattr(self.training_loader, 'last_waiting_time') else 0,
                            'data_move': profiler_data_move.elapsed,
                            'inference': profiler_inference.elapsed,
                            'loss': profiler_loss.elapsed,
                            'backward': profiler_backward.elapsed,
                            'optimizer': profiler_optimizer.elapsed,
                        }
                    }

                    for logger in self.loggers:
                        logger(inference_data)

            if validate:
                assert self.validation_loader, "Trying to validate without validation loader."
                if self.validation_loader:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_index, (x, y) in enumerate(self.validation_loader):
                            with Profiler() as profiler_data_move:
                                x = x.to(self.device)
                                y = y.to(self.device)

                            with Profiler() as profiler_inference:
                                pred = self.model(x)

                            with Profiler() as profiler_loss:
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
                                'lr_scheduler': self.lr_scheduler,
                                'profiles': {
                                    'batch_load': self.validation_loader.last_waiting_time if hasattr(self.validation_loader, 'last_waiting_time') else 0,
                                    'data_move': profiler_data_move.elapsed,
                                    'inference': profiler_inference.elapsed,
                                    'loss': profiler_loss.elapsed,
                                }
                            }

                            for logger in self.loggers:
                                logger(inference_data)

            if self.lr_scheduler:
                self.lr_scheduler.step()


