from itertools import product
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
        self.collector = Collector(device)

        # Initial data collection on completely untrained model.
        self.collector.collect(model=self.model,
                               dataloader=self.dataloaders.train,
                               val_dataloader=self.dataloaders.validation,
                               criterion=self.criterion)

    def train(self, model_params, optimizer_params):
        self.setup(model_params, optimizer_params)

        for epoch in range(self.epochs):
            self.model.train()
            for i, batch in enumerate(self.dataloader):
                x, y = batch
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()

            self.collector.collect(model=self.model,
                                   dataloader=self.dataloader,
                                   val_dataloader=self.validation_dataloader,
                                   criterion=self.criterion)

class HyperParameterSearch:
    def __init__(self, trainer, optimizer_param_group, model_param_group):
        self.trainer = trainer
        self.optimizer_param_group = optimizer_param_group
        self.model_param_group = model_param_group
        self.results = {}

        
    def search(self):
        for optimizer_params in self.optimizer_param_group.as_kwargs:
            for model_params in self.model_param_group.as_kwargs:
                self.trainer.train(optimizer_params, model_params)
                hyper_params = {**optimizer_params, **model_params}
                self.results[hyper_params] = self.trainer.collector

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