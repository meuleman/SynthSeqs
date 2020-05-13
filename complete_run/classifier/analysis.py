import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

from utils.constants import (
    TRAIN,
    VALIDATION,
)


TRUES = 'trues'
PREDS = 'preds'


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

        self.device = device
    
    @torch.no_grad()
    def collect(self, model, dataloader, val_dataloader, criterion):
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

                loss.append(criterion(preds, comps))

                all_trues = torch.cat([all_trues, comps.cpu()], dim=0)
                all_preds = torch.cat([all_preds, preds.cpu()], dim=0)
            
            self.loss_history[label].append(sum(loss) / len(loss))
            self.predictions_history[label].append({
                TRUES: all_trues.numpy(),
                PREDS: all_preds.numpy(),
            })

    @property
    def epochs(self):
        return len(self.loss_history[TRAIN])


class Evaluator:
    def __init__(self, collector, output_dir):
        self.collector = collector
        self.output_dir = output_dir

    def plot_loss(self, filename):
        history = self.collector.loss_history

        plt.figure(figsize=(15, 10))
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        xs = list(range(self.collector.epochs))

        for label in history.keys():
            plt.plot(xs, history[label], label=label)

        plt.legend()
        plt.savefig(self.output_dir + filename)
        plt.close()

    def plot_confusion(self, filename):
        history = self.collector.predictions_history

        for label in history.keys():
            final = history[label][-1]
            y_trues = final[TRUES]
            y_preds = final[PREDS].argmax(axis=1)

            cm = confusion_matrix(y_trues, y_preds)

            # TODO: Leaving out normalization for now to see raw confusion mat.
            # cm = cm / cm.sum(axis=1).reshape(16, 1)
            # cm = np.around(cm, 2)

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

            plt.savefig(self.output_dir + filename)
            plt.close()

