import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from utils.constants import (
    TOTAL_CLASSES,
    TRAIN,
    VALIDATION,
)

from .constants import (
    CONFUSION,
    EPOCH_TIME,
    F1_SCORE,
    LOSS,
    LOSS_DIFF,
    PREDS,
    PRECISION,
    RECALL,
    SEARCH_COLUMN_SCHEMA,
    TRUES,
)


class Collector:
    def __init__(self, device):
        self.loss_history = {
            TRAIN: [],
            VALIDATION: [],
        }

        self.predictions_history = {
            TRAIN: [],
            VALIDATION: [],
        }

        self.epoch_times = []

        self.device = device
    
    @torch.no_grad()
    def collect(self,
                model,
                dataloader,
                val_dataloader,
                criterion,
                epoch_time=None):
        model.eval()
        loaders = {TRAIN: dataloader, VALIDATION: val_dataloader}

        for label in loaders.keys():
            all_preds = torch.tensor([])
            all_trues = torch.LongTensor([])
            loss = []

            for batch in loaders[label]:
                seqs, comps = batch
                seqs = seqs.transpose(-2, -1).to(self.device)
                comps = comps.to(self.device)
                preds = model(seqs)

                loss.append(criterion(preds, comps).item())

                all_trues = torch.cat([all_trues, comps.cpu()], dim=0)
                all_preds = torch.cat([all_preds, preds.cpu()], dim=0)
            
            self.loss_history[label].append(sum(loss) / len(loss))
            self.predictions_history[label].append({
                TRUES: all_trues.numpy(),
                PREDS: all_preds.numpy(),
            })
        
        if epoch_time:
            self.epoch_times.append(epoch_time)

    @property
    def epochs(self):
        return len(self.loss_history[TRAIN])


class Evaluator:
    def __init__(self, collector):
        self.collector = collector

        self.metric_funcs = {
            PRECISION: precision_score,
            RECALL: recall_score,
            F1_SCORE: f1_score,
            CONFUSION: confusion_matrix,
        }
        self.classes = np.arange(TOTAL_CLASSES)

    def _final_clean_preds(self, label):
        final = self.collector.predictions_history[label][-1]
        return final[TRUES], final[PREDS].argmax(axis=1)

    def _sklearn_metrics(self, metric, label, **kwargs):
        y_trues, y_preds = self._final_clean_preds(label)
        metric_func = self.metric_funcs[metric]
        return metric_func(y_trues,
                           y_preds,
                           labels=self.classes,
                           **kwargs)

    def final_loss(self, label):
        return self.collector.loss_history[label][-1]

    def loss_diff(self):
        return self.final_loss(VALIDATION) - self.final_loss(TRAIN)

    def precision(self, label, average='macro'):
        return self._sklearn_metrics(PRECISION, label, average=average)

    def recall(self, label, average='macro'):
        return self._sklearn_metrics(RECALL, label, average=average)
    
    def f1_score(self, label, average='macro'):
        return self._sklearn_metrics(F1_SCORE, label, average=average)

    def confusion_matrix(self, label, normalize=None):
        return self._sklearn_metrics(CONFUSION, label, normalize=normalize)

    def mean_epoch_time(self):
        times = self.collector.epoch_times
        return sum(times) / len(times)

    def _plot_loss_and_stats(self, ax):
        history = self.collector.loss_history

        xs = list(range(self.collector.epochs))
        ys = []
        for label in history.keys():
            ax.plot(xs, history[label], label=label, linewidth=6)
            ys.append(history[label][-1])

        x = xs[-1]
        ymin, ymax = min(ys), max(ys)
        diff = ymax - ymin
        ax.vlines(x, ymin, ymax, linestyles='dashed')
        ax.text(x,
                ymax + .005,
                'Diff: {:.3g}'.format(diff),
                fontsize=24,
                bbox=dict(facecolor='white', alpha=0.5),
                ha='right')

        ax.set_ylabel('Cross entropy loss', fontsize=30)
        ax.set_xlabel('Epochs', fontsize=30)
        ax.set_title("Classifier's cross entropy loss during training",
                     fontsize=40)
        ax.tick_params(labelsize=24)
        ax.legend(fontsize=26)

    def _plot_confusion(self, ax, label, normalize):
        cm = self.confusion_matrix(label, normalize=normalize)

        ax.imshow(cm, cmap='Blues')

        labelsx = [item.get_text() for item in ax.get_xticklabels()]
        labelsy = [item.get_text() for item in ax.get_yticklabels()]
        labelsx[1:-2] = np.arange(1, 17, 2)
        labelsy[1:-2] = np.arange(1, 17, 2)
        ax.set_xticklabels(labelsx, fontsize=24)
        ax.set_yticklabels(labelsy, fontsize=24)
        ax.set_title(f'Confusion matrix of {label} predictions', fontsize=40)
        ax.set_xlabel('Predicted label', fontsize=30)
        ax.set_ylabel('True label', fontsize=30)
        for i in range(16):
            for j in range(16):
                ax.text(j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="orange",
                        fontsize=12,
                        fontweight='heavy')

    def plot_loss_and_stats(self, filename, output_dir):
        _, ax = plt.subplots(figsize=(15, 10))
        self._plot_loss_and_stats(ax)
        plt.savefig(output_dir + filename)

    def plot_confusion(self, filename, output_dir, label, normalize=None):
        _, ax = plt.subplots(figsize=(15, 15))
        self._plot_confusion(ax, label, normalize)
        plt.savefig(output_dir + filename)
    
    def plot_for_search(self, filename, output_dir, label, normalize=None):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        self._plot_loss_and_stats(ax1)
        self._plot_confusion(ax2, label, normalize)
        plt.savefig(output_dir + filename)


class MotifMatch:
    def __init__(self, pwm):
        self.pwm = pwm
        assert self.pwm.shape[0] == 4, f'PWM has shape {self.pwm.shape}'

    def scan(self, sequences):
        num_sequences = len(sequences)
        len_pwm = self.pwm.shape[1]
        num_strides = sequences.shape[2] - len_pwm + 1
        out = np.zeros((num_sequences, num_strides))

        for i in range(num_strides):
            end = i + len_pwm
            out[:, i] = np.tensordot(sequences[:, :, i:end], self.pwm)

        return out

