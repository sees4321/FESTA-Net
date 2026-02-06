import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as opt

from torch.utils.data import DataLoader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OPT_DICT = {'Adam':opt.Adam,
            'AdamW':opt.AdamW,
            'SGD':opt.SGD}

def ManualSeed(seed:int, deterministic=True):
    # random seed 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    r"""
    Args:
        patience (int): Number of epochs to wait for improvement before stopping. Default: 3
        delta (float): Minimum change in the monitored metric to qualify as an improvement. Default: 0.0
        mode (str): One of {'min', 'max'}. In 'min' mode, training stops when the metric stops 
                    decreasing. In 'max' mode, it stops when the metric stops increasing. Default: 'min'
        verbose (bool): If True, prints a message when early stopping is triggered. Default: True
    """
    def __init__(self, model, patience=3, delta=0.0, mode='min', verbose=False):

        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        self.model = model
        self.epoch = 0

    def __call__(self, score, epoch):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                torch.save(self.model.state_dict(), f'best_model.pth')
                self.epoch = epoch
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                torch.save(self.model.state_dict(), f'best_model.pth')
                self.epoch = epoch
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False

def trainer(
        model:nn.Module, 
        train_loader:DataLoader, 
        val_loader:DataLoader, 
        num_epoch:int, 
        optimizer_name:str, 
        learning_rate:str, 
        early_stop:EarlyStopping = None,
        min_epoch:int = 0,
        exlr_on:bool = False,
        num_classes:int = 1
        ):
    assert num_classes in [1,3], 'num classes must be 1 (binary) or 3 (n-back)'

    criterion = nn.BCELoss() if num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = OPT_DICT[optimizer_name](model.parameters(), lr=float(learning_rate))
    exlr = opt.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tr_acc, tr_loss = [], []
    vl_acc, vl_loss = [], []
    train_total, val_total = 0, 0
    early_stopped = False

    for epoch in range(num_epoch):
        model.train()
        train_loss, train_correct = 0.0, 0
        for i, data in enumerate(train_loader, 0):
            x, z, y = data
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = torch.squeeze(model(x, z))
            loss = criterion(pred, y.float()) if num_classes == 1 else criterion(pred, y)
            loss.backward()
            optimizer.step()

            predicted = (pred > 0.5).int() if num_classes == 1 else torch.argmax(pred, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            train_loss += loss.item()
        if exlr_on: exlr.step()
        tr_loss.append(round(train_loss/len(train_loader), 4))
        tr_acc.append(round(100 * train_correct / train_total, 4))

        if early_stop:
            with torch.no_grad():
                model.eval()
                val_loss, val_correct = 0.0, 0 
                for i, data in enumerate(val_loader, 0):
                    x, z, y = data
                    x = x.to(DEVICE)
                    z = z.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    pred = torch.squeeze(model(x, z))
                    predicted = (pred > 0.5).int() if num_classes == 1 else torch.argmax(pred, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()
                    loss = criterion(pred, y.float()) if num_classes == 1 else criterion(pred, y)
                    val_loss += loss.item()

                val_loss = round(val_loss/len(val_loader), 4)
                val_acc = round(100 * val_correct / val_total, 4)
                vl_loss.append(val_loss)
                vl_acc.append(val_acc)

                if epoch > min_epoch: 
                    if early_stop.mode == 'min':
                        early_stop(val_loss, epoch)
                    else:
                        early_stop(val_acc, epoch)
                if early_stop.early_stop:
                    early_stopped = True
                    break  
    if not early_stopped and early_stop:
        torch.save(model.state_dict(), f'best_model.pth')
    return tr_acc, tr_loss, vl_acc, vl_loss

def tester(model:nn.Module, tst_loader:DataLoader, num_classes:int=1):
    assert num_classes in [1,3], 'num classes must be 1 (binary) or 3 (n-back)'
    total = 0
    correct = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, z, y in tst_loader:
            x = x.to(DEVICE)
            z = z.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x, z)
            pred = torch.squeeze(pred)
            predicted = (pred > 0.5).int() if num_classes == 1 else torch.argmax(pred, 1)
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets