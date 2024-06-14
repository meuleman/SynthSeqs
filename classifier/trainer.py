from itertools import product
import pandas as pd
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils.constants import CLASSIFIER, TRAIN, VALIDATION
from utils.data import DHSDataLoader
from utils.net import weights_init

from .analysis import Collector, Evaluator
from .constants import (
    EPOCH_TIME,
    F1_SCORE,
    LOSS,
    LOSS_DIFF,
    PRECISION,
    RECALL,
    SEARCH_COLUMN_SCHEMA,
)


class ClassifierTrainer:
    def __init__(self,
                 epochs,
                 batch_size,
                 model,
                 device,
                 data_dir):
        self.device = device
        self.model_type = model
        self.criterion = CrossEntropyLoss().to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_dir = data_dir

    def setup(self, model_params, optimizer_params):
        self.model = self.model_type(**model_params).to(self.device)
        self.model.apply(weights_init)
        self.opt = Adam(self.model.parameters(), **optimizer_params)

        self.dataloaders = DHSDataLoader(self.batch_size,
                                         CLASSIFIER,
                                         self.data_dir)
        self.collector = Collector(self.device)

        # Initial data collection on completely untrained model.
        self.collector.collect(model=self.model,
                               dataloader=self.dataloaders.train,
                               val_dataloader=self.dataloaders.validation,
                               criterion=self.criterion,
                               epoch_time=None)

    def train(self, model_params, optimizer_params):
        self.setup(model_params, optimizer_params)

        for _ in range(self.epochs):
            start = time.time()
            self.model.train()
            for i, batch in enumerate(self.dataloaders.train):
                x, y = batch
                x = x.float().transpose(-2, -1).to(self.device)
                y = y.long().to(self.device)
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()

            elapsed = time.time() - start
            self.collector.collect(model=self.model,
                                   dataloader=self.dataloaders.train,
                                   val_dataloader=self.dataloaders.validation,
                                   criterion=self.criterion,
                                   epoch_time=elapsed)

    def plot(self, figure_dir):
        evaluator = Evaluator(self.collector)
        evaluator.plot_loss_and_stats('loss.png', figure_dir)
        evaluator.plot_confusion('confusion.png',
                                 figure_dir,
                                 VALIDATION,
                                 normalize='true')

    def save(self, filename, model_dir):
        torch.save(self.model.state_dict(), model_dir + filename)


class HyperParameterSearch:
    def __init__(self,
                 trainer,
                 model_param_group,
                 optimizer_param_group,
                 plot_dir=None):
        self.trainer = trainer
        self.model_param_group = model_param_group
        self.optimizer_param_group = optimizer_param_group

        self.df = pd.DataFrame(columns=SEARCH_COLUMN_SCHEMA)
        self.plot_dir = plot_dir
        
    @property
    def dataframe(self):
        return self.df

    def search(self):
        total_models = (
            len(self.model_param_group) * len(self.optimizer_param_group)
        )
        trained = 0
        for model_params in self.model_param_group.as_kwargs:
            for optimizer_params in self.optimizer_param_group.as_kwargs:
                self.trainer.train(model_params, optimizer_params)

                hyper_params = {**model_params, **optimizer_params}
                self.evaluate(hyper_params, self.trainer.collector)

                trained += 1
                print(f'Completed training {trained} of {total_models} models')

    def evaluate(self, hyper_params, collector):
        metrics = {}
        evaluator = Evaluator(collector)

        def col(label, metric):
            return f'{label}_{metric}'

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

        if self.plot_dir:
            filename = '_'.join(x + str(y) for x, y in hyper_params.items())
            bad_chars = ['.', ' ', '(', ')']
            for char in bad_chars:
                filename = filename.replace(char, '')
            filename = filename.replace(',', '-')
            evaluator.plot_for_search(filename,
                                      self.plot_dir,
                                      VALIDATION,
                                      normalize='true')

    def save(self, output_dir, filename):
        self.df.to_csv(output_dir + filename)


class ParameterGroup:
    def __init__(self, params):
        self.params = params
    
    @property
    def as_args(self):
        return product(*self.params.values())

    @property
    def as_kwargs(self):
        labels = self.params.keys()
        for group in self.as_args:
            yield dict(zip(labels, group))

    def __len__(self):
        total = 1
        for v in self.params.values():
            total *= len(v)
        return total
