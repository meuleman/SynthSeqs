import torch
import torch.nn as nn
import torch.optim as optim
import data_helper
import utils
import numpy as np
from gen_models import resnet_block
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

class hot_dog_not_hot_dog(nn.Module):
    def __init__(self):
        super(hot_dog_not_hot_dog, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(11, 4), stride=1, padding=(5, 0), bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(kernel_size=(20, 1))
        self.fc1 = nn.Linear(1280, 200)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 1)
        self.sig = nn.Sigmoid()

    def forward(self, seq):
        h = self.conv1(seq.view(-1, 1, 100, 4))
        h = self.lrelu1(h)
        h = self.drop1(h)
        h = self.bn1(h)
        h = self.pool1(h).view(-1, 1280)
        h = self.fc1(h)
        h = self.lrelu2(h)
        h = self.drop2(h)
        h = self.fc2(h)
        out = self.sig(h).squeeze()
        return out

class resnet_hot_dog_1d(nn.Module):
    def __init__(self):
        super(resnet_hot_dog_1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 100, kernel_size=1, stride=1),
            nn.ReLU(True),
            resnet_block(100, 100, 5, spec_norm=False),
            nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            nn.MaxPool1d(20)
        )
        self.fc = nn.Linear(500, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.net(x).view(-1, 500)
        fc = self.fc(h)
        out = self.sig(fc)
        return out.squeeze()


class resnet_all(nn.Module):
    def __init__(self, drop=0.2):
        super(resnet_all, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 100, kernel_size=11, stride=1, padding=5),
            nn.ReLU(True),
            #nn.BatchNorm1d(320),
            resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            nn.MaxPool1d(25)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(True),
            #nn.Dropout(drop),
            nn.BatchNorm1d(100),
            nn.Linear(100, 16),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.net(x).view(-1, 400)
        out = self.fc_net(h)
        return out.squeeze()


class tmp(nn.Module):
    def __init__(self, drop=0.2):
        super(tmp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 512, kernel_size=17, stride=1, padding=8),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            #resnet_block(100, 100, 5, spec_norm=False),
            #nn.Dropout(0.5),
            nn.MaxPool1d(100)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.BatchNorm1d(100),
            nn.Linear(100, 16),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.net(x).view(-1, 512)
        out = self.fc_net(h)
        return out.squeeze()



class classifier_trainer():
    def __init__(self, epochs, bs, lr, b1=0.9, b2=0.999, drop=0.0, c=None):
        self.dataloader, self.test_dataloader = data_helper.get_the_dataloaders(bs, binary_class=c, weighted_sample=True, one_dim=True, data_type='strong', rc=True)
        self.c = c
        if self.c is None:
            self.model = tmp(drop).to("cuda") ##
            self.criterion = nn.CrossEntropyLoss().to("cuda")
        else:
            self.model = resnet_hot_dog_1d().to("cuda")
            self.criterion = nn.BCELoss().to("cuda")
        self.opt = optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        self.epochs = epochs
        self.train_hist = []
        self.train_hist_test = []
        self.model.apply(utils.weights_init)
        self.count_parameters()

    def accuracy(self, output, target):
        pred = output >= 0.5
        truth = target >= 0.5
        acc = pred.eq(truth).sum().item() / target.numel()
        return acc

    def count_parameters(self):
        #return print(p.numel() for p in self.model.parameters() if p.requires_grad)
        for p in self.model.parameters():
            if p.requires_grad:
                print(p.numel())

    def train(self):
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.dataloader):
                x, y = batch
                x = x.float().to("cuda")
                if self.c is not None:
                    y = y.float().to("cuda")
                else:
                    y = y.long().to("cuda")
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()
                #if i % 100 == 0:
                #    self.train_hist.append(loss.item())

                if i % 3000 == 0:
                    self.model.eval()
                    #a = np.array([])
                    #for j, b in enumerate(self.test_dataloader):
                    #    p = self.model(b[0].float().to("cuda"))
                    #    a = np.append(a, self.accuracy(p.detach(), b[1].float().to("cuda").detach()))
                    #conf_low = a.mean() - ((1.96 * np.std(a))/np.sqrt(len(a)))
                    #conf_high = a.mean() + ((1.96 * np.std(a))/np.sqrt(len(a)))
                    #train_a = self.accuracy(pred.detach(), y.detach())
                    y_true, y_pred, test_loss = self.get_preds()
                    self.train_hist_test.append(test_loss.item())
                    self.train_hist.append(loss.item())
                    if self.c is not None:
                        ap = self.precision_recall(y_true, y_pred)
                    else:
                        ap = None
                    cm = self.confusion(y_true, y_pred)
                    np.set_printoptions(precision=2, linewidth=100, suppress=True)
                    print('Epoch: {0}, Iter: {1}, Loss: {2}, TestLoss: {3}, AP: {4}'.format(
                          epoch, i, loss.item(), test_loss.item(), ap)
                    )
                    print('Confusion Matrix: ')
                    print(cm)
                    print("")
                    if epoch == self.epochs-1:
                        np.save("/home/pbromley/generative_dhs/notebooks/cm.npy", cm)
                    self.model.train(True)
        return self.model

    def plot_loss(self):
        x = range(len(self.train_hist))
        plt.figure(figsize=(10, 10))
        plt.plot(x, self.train_hist)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        if self.c is not None:
            plt.savefig(('/home/pbromley/generative_dhs/loss_plots/aug-%d.png' % self.c))
        else:
            plt.savefig(('/home/pbromley/generative_dhs/loss_plots/aug.png'))
        plt.close()

    def get_preds(self):
        y_true = np.zeros(len(self.test_dataloader.dataset))
        if self.c is not None:
            y_pred = np.zeros(len(self.test_dataloader.dataset))
        else:
            y_pred = np.zeros((len(self.test_dataloader.dataset), 16))
        self.model.eval()
        loss = 0
        for i, batch in enumerate(self.test_dataloader):
            bs = batch[0].size()[0]
            y_true[i*bs:(i+1)*bs] = batch[1].numpy()
            pred = self.model(batch[0].float().to("cuda")).detach()
            y_pred[i*bs:(i+1)*bs] = pred.cpu().numpy()
            
            if self.c is not None:
                loss += self.criterion(pred, batch[1].float().to("cuda"))
            else:
                loss += self.criterion(pred, batch[1].long().to("cuda"))
        loss = loss / i
        return y_true, y_pred, loss

    def precision_recall(self, y_true, y_pred, plot=False):
        p, r, thresh = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        if plot:
            plt.step(r, p, color=(utils.get_dhs_colors())[self.c], where='post')
            plt.fill_between(r, p, step='post', color=(utils.get_dhs_colors())[self.c])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("AUPRC for component {0}, Avg. Precision: {1:0.2f}".format(self.c+1, ap))
            plt.savefig(('/home/pbromley/generative_dhs/auprc/strong-%d.png' % self.c))
            plt.close()
        return ap 

    def confusion(self, y_true, y_pred):
        if self.c is not None:
            cm = confusion_matrix(y_true, (y_pred >= 0.5).astype(int))
        else:
            cm = confusion_matrix(y_true, y_pred.argmax(axis=1))
            cm = np.around(cm/cm.sum(axis=1).reshape(16, 1), 2)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(cm, cmap='Blues')
            labelsx = [item.get_text() for item in ax.get_xticklabels()]
            labelsy = [item.get_text() for item in ax.get_yticklabels()]
            labelsx[1:-2] = np.arange(1, 17, 2)
            labelsy[1:-2] = np.arange(1, 17, 2)
            ax.set_xticklabels(labelsx)
            ax.set_yticklabels(labelsy)
            for i in range(16):
                for j in range(16):
                    text = ax.text(j, i, cm[i, j], ha="center", va="center", color="orange")
            plt.savefig(('/home/pbromley/generative_dhs/auprc/confusion.png'))
            plt.close()
        return cm



if __name__ == "__main__":
    ### BINARY ###
    #for i in range(16):
        #print("Training component " + str(i))
        #trainer = classifier_trainer(20, 128, 0.0003, c=i)
        #model = trainer.train()
        #y_true, y_pred, _ = trainer.get_preds()
        #trainer.precision_recall(y_true, y_pred, plot=True)
        #trainer.plot_loss(i)
        #torch.save(model.state_dict(), "/home/pbromley/generative_dhs/saved_models/classifiers/resnet_12.pth")

    ### ALL ###
    lrs = np.arange(0.001, 0.009, 0.0008)
    beta1s = np.arange(0.7, 0.9, 0.1)
    beta2s = np.arange(0.69, 1.0, 0.1)
    drops = np.arange(0.0, 0.8, 0.3)

    #for i in range(len(lrs)):
    #    for j in range(len(beta1s)):
    #        for k in range(len(beta2s)):
    #            for l in range(len(drops)):
    #                trainer = classifier_trainer(10, 256, lrs[i], b1=beta1s[j], b2=beta2s[k], drop=drops[l], c=None)
    #                model = trainer.train()
    #                print(str(lrs[i]), end="\t")
    #                print(str(beta1s[j]), end="\t")
    #                print(str(beta2s[k]), end="\t")
    #                print(str(drops[l]), end="\t")
    #                print(trainer.train_hist[-1], end="\t")
    #                print(trainer.train_hist_test[-1])
    
    #trainer = classifier_trainer(50, 256, 0.0018, b1=0.9, b2=0.99, drop=0.2, c=None)
    trainer = classifier_trainer(50, 256, 0.0018, b1=0.9, b2=0.99, drop=0.2, c=None)
    model = trainer.train()
    #torch.save(model.state_dict(), "/home/pbromley/generative_dhs/saved_models/classifiers/aug.pth")
