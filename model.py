from torch import nn, device, tensor, no_grad
import pandas as pd
import numpy as np
import torch

class model(nn.Module):
    
    def __init__(self, dIn, nN, dO, crit, optim, optimArgs, epochs):
        self.din = dIn
        self.nodes = nN
        self.dout = dO
        self.dev = device('cpu')
        super(model, self).__init__()
        self.l1 = nn.Linear(self.din, self.nodes)
        self.l2 = nn.Linear(self.nodes, self.dout)
        self.criterion = crit
        self.optimizer_ = optim
        self.optim_args = optimArgs
        self.optimizer = None
        self.makeoptimizer()
        self.epochs = epochs
        
    def makeoptimizer(self):
        self.optimizer = self.optimizer_(self.parameters(), **self.optim_args)
        
    def makesureitstorch(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.FloatTensor(x).to(self.dev)
        return x
    
    def forward(self, x):
        x = self.makesureitstorch(x)
        relu = self.l1(x).clamp(min=0)
        return self.l2(relu)
    
    def predict(self, x):
        with no_grad():
            return np.array(self.forward(x).detach())
        
    def learn(self, x, y):
        x = self.makesureitstorch(x)
        y = self.makesureitstorch(y)
        for e in range(self.epochs):
            pred = self.forward(x)
            loss = self.criterion(pred, y)
            loss_val = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_val, self.score, self.state_dict()
    
    @property
    def score(self):
        norm = 0
        l1 = self.l1.weight.detach().numpy()
        l2 = self.l2.weight.detach().numpy()
        norm += np.linalg.norm(l1)
        norm += np.linalg.norm(l2)
        return norm
    
    def saveto(self, fname_no_ext):
        torch.save(self.state_dict(), fname_no_ext + '.pt')
        return fname
    
    def loadfrom(self, sd):
        return 
        
