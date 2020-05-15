from itertools import product
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils.constants import CLASSIFIER
from utils.data import DHSDataLoader
from utils.net import weights_init

from .analysis import Collector


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

        for epoch in range(self.epochs):
            start = time.time()
            self.model.train()
            for i, batch in enumerate(self.dataloaders.train):
                x, y = batch
                x = x.float().to(self.device)
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

class HyperParameterSearch:
    def __init__(self, trainer, model_param_group, optimizer_param_group): 
        self.trainer = trainer
        self.model_param_group = model_param_group
        self.optimizer_param_group = optimizer_param_group
        self.results = []

        
    def search(self):
        total_models = (
            len(self.model_param_group) * len(self.optimizer_param_group)
        )
        trained = 0
        for model_params in self.model_param_group.as_kwargs:
            for optimizer_params in self.optimizer_param_group.as_kwargs:
                self.trainer.train(model_params, optimizer_params)
                hyper_params = {**model_params, **optimizer_params}
                self.results.append((hyper_params, self.trainer.collector))
                trained += 1
                print(f'Completed training {trained} of {total_models} models')

        return self.results


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
