import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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
                seqs = seqs.to(self.device)
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

    def plot_loss(self, filename, output_dir):
        history = self.collector.loss_history

        plt.figure(figsize=(15, 10))
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        xs = list(range(self.collector.epochs))

        for label in history.keys():
            plt.plot(xs, history[label], label=label)

        plt.legend()
        plt.savefig(output_dir + filename)
        plt.close()

    def plot_confusion(self, label, filename, output_dir, normalize=None):
        cm = self.confusion_matrix(label, normalize=normalize)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cm, cmap='Blues')

        # labelsx = [item.get_text() for item in ax.get_xticklabels()]
        # labelsy = [item.get_text() for item in ax.get_yticklabels()]
        # labelsx[1:-2] = np.arange(1, 17, 2)
        # labelsy[1:-2] = np.arange(1, 17, 2)
        # ax.set_xticklabels(labelsx)
        # ax.set_yticklabels(labelsy)
        for i in range(16):
            for j in range(16):
                text = ax.text(j,
                                i,
                                cm[i, j],
                                ha="center",
                                va="center",
                                color="orange")

        plt.savefig(output_dir + filename)
        plt.close()


class SearchEvaluator:
    def __init__(self, results):
        self.results = results
        self.df = pd.DataFrame(columns=SEARCH_COLUMN_SCHEMA)

    @property
    def dataframe(self):
        return self.df

    def build_eval_df(self):
        def col(label, metric):
            return f'{label}_{metric}'

        for hyper_params, collector in self.results:
            metrics = {}
            evaluator = Evaluator(collector)
            # Some metrics have both train and validation values.
            for label in [TRAIN, VALIDATION]:
                metrics[col(label, LOSS)] = evaluator.final_loss(label)
                metrics[col(label, PRECISION)] = (
                    evaluator.precision(label, average='macro')
                )
                metrics[col(label, RECALL)] = (
                    evaluator.recall(label, average='macro')
                )
                metrics[col(label, F1_SCORE)] = (
                    evaluator.f1_score(label, average='macro')
                )

            metrics[LOSS_DIFF] = evaluator.loss_diff()
            metrics[EPOCH_TIME] = evaluator.mean_epoch_time()

            row = {**hyper_params, **metrics}

            self.df = self.df.append(row, ignore_index=True)

    def save(self, output_dir, filename):
        self.df.to_csv(output_dir + filename)

    def evaluate(self, output_dir, filename):
        self.build_eval_df()
        self.save(output_dir, filename)

