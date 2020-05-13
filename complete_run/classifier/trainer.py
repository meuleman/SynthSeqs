from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from utils.constants import CLASSIFIER
from utils.data import DHSDataLoader
from utils.net import weights_init


class ClassifierTrainer:
    def __init__(self,
                 epochs,
                 batch_size,
                 optimizer_params,
                 model_params,
                 model,
                 collector,
                 data_dir,
                 device):
        self.device = device

        dataloaders = DHSDataLoader(batch_size, CLASSIFIER, data_dir)
        self.dataloader = dataloaders.train
        self.validation_dataloader = dataloaders.validation

        self.model = model(*model_params).to(self.device)
        self.criterion = CrossEntropyLoss().to(self.device)
        self.opt = Adam(self.model.parameters(), **optimizer_params)
        self.epochs = epochs

        self.model.apply(weights_init)

        self.collector = collector

        self.collector.collect(model=self.model,
                               dataloader=self.dataloader,
                               val_dataloader=self.validation_dataloader,
                               criterion=self.criterion)

    def train(self):
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
