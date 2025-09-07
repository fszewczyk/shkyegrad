import wandb
import torch
import numpy as np
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import pathlib
import json
import secrets
from datetime import datetime
import time

class InferenceLogger:
    def __init__(self):
        pass

    def __call__(self, inference_data):
        self.update(inference_data)

class LearningRateLogger(InferenceLogger):
    def __init__(self, log_every_epoch=1, log_every_batch=math.inf, log_console=True, log_wandb=False):
        self.log_every_epoch = log_every_epoch
        self.log_every_batch = log_every_batch

        self.log_console = log_console
        self.log_wandb = log_wandb

        self.__previous_epoch = -1

    def update(self, inference_data):
        if not inference_data['train']:
            return

        data = None
        new_epoch = self.__previous_epoch != inference_data['epoch']
        if new_epoch and inference_data['epoch'] % self.log_every_epoch == 0:
            data = {'epoch': inference_data['epoch']}
        if inference_data['global_batch_index'] % self.log_every_batch == 0:
            data = {'batch': inference_data['global_batch_index']}
        
        if data:
            info_string = f"Epoch: {inference_data['epoch'] + 1} Batch: {inference_data['batch_index'] + 1}"
            for i, param_group in enumerate(inference_data['optimizer'].param_groups):
                data[f'lr/group-{i}'] = param_group['lr']
                info_string += f"\n\t LR Group {i}: {param_group['lr']}"

            if self.log_wandb:
                wandb.log(data)

        self.__previous_epoch = inference_data['epoch']

class ProfileLogger(InferenceLogger):
    def __init__(self, log_every_batch, log_console=True, log_wandb=False):
        self.log_every_batch = log_every_batch

        self.start_time = None
        self.start_batch = None
        self.total_samples = 0
        self.current_mode = None

        self.last_timings = {
            'train': {},
            'valid': {}
        }

        self.log_console = log_console
        self.log_wandb = log_wandb

    def update(self, inference_data):
        mode = 'train' if inference_data['train'] else 'valid'
        if self.current_mode != mode:
            self.current_mode = mode
            self.start_time = None
            self.total_samples = 0
            self.last_timings = {
                'train': {},
                'valid': {}
            }

            return

        if self.start_time is None:
            self.start_time = time.perf_counter()
            self.start_batch = inference_data['global_batch_index']

        self.total_samples += inference_data['input'].size(0)
        
        timing_data = self.last_timings[mode]
        for name, timing in inference_data['profiles'].items():
            sum_timings = timing_data.get(name, 0)
            sum_timings += timing
            timing_data[name] = sum_timings

        if (inference_data['global_batch_index'] - self.start_batch + 1) % self.log_every_batch == 0:
            if self.log_console:
                print(f"Epoch {inference_data['epoch']}, Batches {inference_data['global_batch_index'] - self.log_every_batch}-{inference_data['global_batch_index']}")

            loop_time = self.__log_throughput(inference_data)
            self.__log_profiles(inference_data, loop_time)
            self.last_timings = {
                'train': {},
                'valid': {}
            }
            self.start_time = None
            self.total_samples = 0

    def __log_throughput(self, inference_data):
        end_time = time.perf_counter()
        throughput = self.total_samples / (end_time - self.start_time)
        loop_time = (end_time - self.start_time) / (inference_data['global_batch_index'] - self.start_batch)
        
        if self.log_console:
            mode = 'Train' if inference_data['train'] else 'Valid'
            info_string = f"\t{mode} Throughput: {throughput:.2f} samples/sec"
            info_string += f"\n\t{mode} Loop Time: {loop_time} sec"
            print(info_string)

        if self.log_wandb:
            mode = 'train' if inference_data['train'] else 'valid'
            wandb.log({
                f'{mode}/throughput': throughput,
                f'{mode}/loop-time': loop_time,
                'batch': inference_data['global_batch_index'],
            })
    
        return loop_time

    def __log_profiles(self, inference_data, loop_time):
        mode = 'train' if inference_data['train'] else 'valid'
        wandb_log = {'batch': inference_data['global_batch_index']}
        profiled_time = 0
        for name, sum_timings in self.last_timings[mode].items():
            mean = sum_timings / self.log_every_batch
            profiled_time += mean
            wandb_log[f'{mode}/timing-{name}'] = mean

            if self.log_console:
                print(f'\t{mode} - {name}: {mean}')

        unprofiled_time = loop_time - profiled_time
        wandb_log[f'{mode}/timing-unprofiled'] = unprofiled_time
        if self.log_console:
            print(f'\tUnprofiled Time: {unprofiled_time}')

        if self.log_wandb:
            wandb.log(wandb_log)


class ArtifactLogger(InferenceLogger):
    def __init__(self, directory: str, log_every_epoch=1):
        self.directory = pathlib.Path(directory)
        self.log_every_epoch = log_every_epoch
        self.__previous_epoch = -1

    def update(self, inference_data):
        if not inference_data['train']:
            return

        new_epoch = self.__previous_epoch != inference_data['epoch']
        if new_epoch and inference_data['epoch'] % self.log_every_epoch == 0:
            model = inference_data['model']
            optimizer = inference_data['optimizer']
            epoch = inference_data['epoch']
            lr_scheduler = inference_data.get('lr_scheduler')

            random_hash = secrets.token_hex(4)

            # Save model state dict
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if lr_scheduler:
                checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
            
            checkpoint_path = self.directory / f"checkpoint_epoch_{epoch}_{random_hash}"
            torch.save(checkpoint, checkpoint_path)

            # Load existing metadata
            metadata_path = self.directory / "metadata.json"
            all_metadata = []
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    try:
                        all_metadata = json.load(f)
                    except json.JSONDecodeError:
                        all_metadata = []
            
            # Add new metadata
            new_metadata = {
                'epoch': epoch,
                'global_batch_index': inference_data['global_batch_index'],
                'path': str(checkpoint_path),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            all_metadata.append(new_metadata)

            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(all_metadata, f, indent=4)

        self.__previous_epoch = inference_data['epoch']

class LossLogger(InferenceLogger):
    def __init__(self, epochs: int, train_batches: int, valid_batches: int = None, log_every_epoch: int = 1, log_every_batch: int = math.inf, log_wandb=False, log_console=True, device='cpu'):
        store_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train = torch.zeros((epochs, train_batches), requires_grad=False, dtype=torch.float32).to(store_device)
        self.valid = None if valid_batches is None else torch.zeros((epochs, valid_batches), requires_grad=False, dtype=torch.float32).to(store_device)
    
        self.log_every_epoch = log_every_epoch
        self.log_every_batch = log_every_batch

        self.train_batches = train_batches
        self.valid_batches = valid_batches

        self.training_mode = True
        self.epoch = 0

        self.log_wandb = log_wandb
        self.log_console = log_console

    def __log_epoch(self):
        train = torch.mean(self.train[self.epoch])
        valid = torch.mean(self.valid[self.epoch]) if self.valid is not None else None
        if self.log_console:
            info_string = f"Epoch {self.epoch + 1}/{self.train.size(0)}"
            info_string += f"\n\tEpoch Train Loss: {train}"
            if self.valid is not None:
                info_string += f"\n\tEpoch Validation Loss: {valid}"
            print(info_string) 

        if self.log_wandb:
            wandb.log({'train/loss_epoch': train, 'epoch': self.epoch})
            if self.valid is not None:
                wandb.log({'valid/loss_epoch': valid, 'epoch': self.epoch})

    def __log_batch(self, inference_data):
        if self.log_console:
            info_string = f"Epoch {self.epoch + 1}/{self.train.size(0)}"
            info_string += f"\n\tBatch {inference_data['batch_index'] + 1}/{self.train_batches if inference_data['train'] else self.valid_batches}"
            if inference_data['train']:
                info_string += f"\n\t Epoch Train Loss: {torch.mean(self.train[inference_data['epoch'], :inference_data['batch_index'] + 1])}"
                info_string += f"\n\t Batch Train Loss: {inference_data['loss']}"
            else:
                info_string += f"\n\t Epoch Validation Loss: {torch.mean(self.valid[inference_data['epoch'], :inference_data['batch_index'] + 1])}"
                info_string += f"\n\t Batch Validation Loss: {inference_data['loss']}"
            print(info_string)

        if self.log_wandb:
            if inference_data['train']:
                wandb.log({'train/loss_batch': inference_data['loss'], 'batch': inference_data['global_batch_index']})
            else:
                wandb.log({'valid/loss_batch': inference_data['loss'], 'batch': inference_data['global_batch_index']})

    def update(self, inference_data):
        new_epoch = False
        if self.epoch != inference_data['epoch']:
            new_epoch = True
        
        if inference_data['train']:
            self.train[inference_data['epoch'], inference_data['batch_index']] = inference_data['loss']
        else:
            self.valid[inference_data['epoch'], inference_data['batch_index']] = inference_data['loss']
 
        if new_epoch and self.epoch % self.log_every_epoch == 0:
            self.__log_epoch()
        elif (inference_data['global_batch_index'] + 1) % self.log_every_batch == 0:
            self.__log_batch(inference_data)
        
        self.epoch = inference_data['epoch']
        self.training_mode = inference_data['train']

    def plot(self, xlabel='Epoch', ylabel='Loss', **kwargs):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        
        ax.plot(np.arange(1, self.train.size(0) + 1), self.train.mean(dim=1), label='Train', **kwargs)

        if self.valid is not None:
            ax.plot(np.arange(1, self.train.size(0) + 1), self.valid.mean(dim=1), label='Validation', **kwargs)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, self.train.size(0))
        ax.grid()
        ax.legend()

        plt.suptitle('Loss over Epochs')
        plt.tight_layout()
        plt.show()

class ClassifierLogger(InferenceLogger):
    @dataclass
    class Prediction:
        x: np.ndarray
        target: str
        prediction: str
        correct: bool
        distribution: list[tuple[float, str]]
        confidence: float

    def __init__(self, classes, index_to_name, epochs, log_every_epoch=1, top_ks=None, log_console=True, log_wandb=False, log_first_valid_batch=0, log_detailed_on_last_epoch=False):
        assert top_ks is None or (len(top_ks) <= 5 and 1 in top_ks)

        self.index_to_name = index_to_name
        self.top_ks = top_ks if top_ks else [1, 3, 5, 10]
        self.classes = classes

        self.train = {
            'confusion_matrix': np.zeros((epochs, classes, classes), dtype=np.uint32),
            'error_rates': {
                k: [[] for _ in range(epochs)]
                for k in self.top_ks
            },
            'predictions': [[] for _ in range(epochs)]
        }
        self.valid = {
            'confusion_matrix': np.zeros((epochs, classes, classes), dtype=np.uint32),
            'error_rates': {
                k: [[] for _ in range(epochs)]
                for k in self.top_ks
            },
            'predictions': [[] for _ in range(epochs)]
        }

        self.valid_exists = False

        self.log_detailed_on_last_epoch = log_detailed_on_last_epoch
        self.log_console = log_console
        self.log_wandb = log_wandb
        self.log_every_epoch = log_every_epoch
        self.log_first_valid_batch = log_first_valid_batch

        self.epochs = epochs
        self.epoch = 0

    def plot_confusion_matrix(self, epoch=-1, figsize=(7, 7)):
        conf_matrix = self.valid['confusion_matrix'][epoch] if self.valid_exists else self.train['confusion_matrix'][epoch]
        confusion_matrix = conf_matrix / np.sum(conf_matrix, axis=1)
        confusion_matrix *= 100

        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        im = ax.imshow(confusion_matrix, cmap='viridis')

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, f"{confusion_matrix[i, j]:.1f}%", ha='center', va='center',
                        color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black')

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(self.index_to_name)), [name for name in self.index_to_name], rotation=45)
        ax.set_yticks(range(len(self.index_to_name)), [name for name in self.index_to_name])
        ax.set_title("Confusion Matrix (%)") 
        plt.colorbar(im, ax=ax, label="Percentage")
        plt.tight_layout()
        plt.show()

    def plot_error_rates(self, ks=None, **kwargs):
        ks = ks if ks else self.top_ks

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        epochs = np.arange(start=1, stop=self.epochs + 1)

        colors = [f'tab:{c}' for c in ['blue', 'orange', 'green', 'red', 'purple']]
        
        for i, (k, error_rates) in enumerate(self.train['error_rates'].items()):
            if k in ks:
                mean_train_error_rate = [100 * sum(er) / len(er) for er in error_rates]
                ax.plot(epochs, mean_train_error_rate, label=f'Train - Top {k}', linestyle='dotted', c=colors[i], **kwargs)

        if self.valid_exists is not None:
            for i, (k, error_rates) in enumerate(self.valid['error_rates'].items()):
                if k in ks:
                    mean_train_error_rate = [100 * sum(er) / len(er) for er in error_rates]
                    ax.plot(epochs, mean_train_error_rate, label=f'Validation - Top {k}', c=colors[i], **kwargs)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error Rate (%)")
        ax.set_ylim(0, None)
        ax.set_xlim(1, self.epochs)
        ax.grid()
        ax.legend()

        plt.suptitle('Error Rate over Epochs')
        plt.tight_layout()
        plt.show() 

    @staticmethod
    def plot_predictions_for_images(predictions: list[Prediction], cmap=None, title='Predictions', top_k=5):
        fig, axs = plt.subplots(2, len(predictions), figsize=(len(predictions) * 1.5, 3), dpi=300)

        if len(predictions) == 1:
            axs = np.array([[axs[0]], [axs[1]]])

        for i, pred in enumerate(predictions):
            im_dims = pred.x.shape[0]
            im = np.squeeze(pred.x)
            cmap = cmap if cmap else 'gray'
            if im_dims == 3:
                im = np.transpose(pred.x, (0, 2))

            axs[0, i].imshow(im, cmap=cmap)
            axs[0, i].axis('off')
            axs[0, i].set_title(f"T: {pred.target}\nP: {pred.prediction}", color='green' if pred.correct else 'red', fontsize=10)

            probs, labels = zip(*pred.distribution)
            probs = probs[:top_k]
            labels = labels[:top_k]
            axs[1, i].bar(range(len(probs)), probs, color='skyblue')
            axs[1, i].set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=6)
            axs[1, i].set_ylim(0, 1)
            axs[1, i].set_ylabel('Prob', fontsize=8)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def __is_logits(dists):
        dist_sums = torch.sum(dists, dim=-1)
        logits = True
        if dists.min() >= 0 and \
            dists.max() <= 1 and \
            torch.allclose(dist_sums, torch.ones_like(dist_sums), atol=1e-3):
            logits = False
        return logits
    
    def __get_top_k_error_rate(self, pred_dists, true_indices, k):
        confidences, pred_indices = torch.topk(pred_dists, k, dim=-1)
        correct = 0.0
        correct = (pred_indices == true_indices.unsqueeze(1)).any(dim=1).float().mean()
        error_rate = 1.0 - correct.item()

        return error_rate
    
    def __log_epoch(self):
        if self.log_console:
            print(f'Epoch {self.epoch + 1}/{self.epochs} Error Rates')

        wandb_log = {}
        for k in self.train['error_rates'].keys():
            error_rates = self.train["error_rates"][k][self.epoch]
            rate = sum(error_rates) / len(error_rates)
            wandb_log[f'train/top-{k}'] = rate
            if self.log_console:
                print(f'\tTop-{k} Train: {rate}')
        
        if self.valid_exists:
            for k in self.valid['error_rates'].keys():
                error_rates = self.valid["error_rates"][k][self.epoch]
                rate = sum(error_rates) / len(error_rates)
                wandb_log[f'valid/top-{k}'] = rate
                if self.log_console:
                    print(f'\tTop-{k} Validation: {rate}')

        if self.log_wandb:
            wandb_log['epoch'] = self.epoch
            wandb.log(wandb_log)

    def __log_prediction(self, inference_data, pred_dists, true_indices):
        data = self.train if inference_data['train'] else self.valid
        pred_confidence, pred_label_indices = torch.max(pred_dists, dim=-1)
        for x, target, pred, pred_dist in zip(inference_data['input'], true_indices, pred_label_indices, pred_dists):
            pred_name = self.index_to_name[pred.item()]
            target_name = self.index_to_name[target.item()]
            named_pred_dist = [
                (conf, self.index_to_name[index]) for index, conf in enumerate(pred_dist.squeeze().cpu())
            ]
            named_pred_dist = sorted(named_pred_dist, key=lambda x: -x[0])

            data['predictions'][inference_data['epoch']].append(
                ClassifierLogger.Prediction(
                    x.detach().cpu().squeeze().numpy(),
                    target_name,
                    pred_name,
                    bool(pred == target),
                    named_pred_dist,
                    named_pred_dist[0][0]
                )
            )

    def update(self, inference_data):
        # Select Correct Data
        self.valid_exists = self.valid_exists or not inference_data['train']
        data = self.train if inference_data['train'] else self.valid
        
        # Make sure output is Softmaxed
        pred_dists = inference_data['prediction']
        true_dists = inference_data['target']
        if ClassifierLogger.__is_logits(pred_dists):
            pred_dists = torch.softmax(pred_dists, dim=-1)

        # Store confusion matrix
        true_label_indices = torch.max(true_dists, dim=-1)[1]
        pred_label_indices = torch.max(pred_dists, dim=-1)[1]
        for true_idx, pred_idx in zip(true_label_indices, pred_label_indices):
            data['confusion_matrix'][inference_data['epoch']][true_idx][pred_idx] += 1
        
        # Measure top-k error rates
        for k in self.top_ks:
            data['error_rates'][k][inference_data['epoch']].append(self.__get_top_k_error_rate(pred_dists, true_label_indices, k))

        # Log detailed prediction
        last_epoch = inference_data['last_epoch']
        log_detailed = (self.valid_exists and not inference_data['train']) or (not self.valid_exists)
        if last_epoch and log_detailed:
            self.__log_prediction(inference_data, pred_dists, true_label_indices)

        # Log metrics
        if inference_data['epoch'] != self.epoch and (inference_data['epoch'] % self.log_every_epoch == 0):
            assert inference_data['epoch'] == self.epoch + 1, f"Some epochs were skipped between {self.epoch} and {inference_data['epoch']}"
            self.__log_epoch()

        self.epoch = inference_data['epoch']
