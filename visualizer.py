import matplotlib.pyplot as plt
import numpy as np
from algorithm import loader
import os
import json

class FineData:
    
    def __init__(self, l, r, n):
        self.l = l
        self.r = r
        self.n = n
        
    def fine1d(self):
        x = np.linspace(self.l, self.r, self.n)
        return x.reshape(self.n, 1)

    def fine2d(self):
        x = (self.l, self.r)
        y = (self.l, self.r)
        data = []
        xpts = np.linspace(x[0], x[1], self.n)
        ypts = sorted(np.linspace(y[0], y[1], self.n), reverse=True)
        for xpt in xpts:
            for ypt in ypts:
                data.append([xpt, ypt])
        data = np.array(data).astype(np.float32)
        return data
    
fd = FineData(-1., 1., 100)
fine1d = fd.fine1d
fine2d = fd.fine2d

class Visualizer:
    
    def __init__(self, name):
        self.path = name + '/'
        self.name = name
        l = loader(self.name)
        loaded = l.load()
        self.data = loaded['data']
        self.state_dicts = loaded['pt']
        self.scores = loaded['scores']
        self.loss_vals = loaded['loss_vals']
        self.choice_i = loaded['choice_i']
        with open(self.name + '/algo.json') as fp:
            self.js = json.loads(fp.read())
        ma = "(" + ", ".join(self.js['model_args']) + ")"
        c, o = self.js['criterion'], self.js['optimizer']
        oa = list(zip(*self.js['optim_args']))
        oa = "{" + ", ".join([f"{a[0]}={a[1]}" for a in oa]) + "}"
        self.title = f"Data: ({self.data.dims}x{self.data.n})\n"
        self.title += f"Architecture: {ma}\nLoss: {c}, E: {self.js['epochs']}"
        self.title += f"\nOptimizer: {o}, {oa}"
        self.mb = loaded['mb']
        
    def loadmodel(self, i):
        m = self.mb()
        m.load_state_dict(self.state_dicts[str(i)])
        m.eval()
        return m
    
    def yield_subsets(self):
        initl = np.where(np.isnan(self.scores[:,[0]]))[0]
        feats, labls = self.data.features, self.data.labels
        # initial training set, no scores
        yield self.loadmodel(0), feats[initl], labls[initl], None, None, None
        for i in range(self.scores.shape[-1]):
            ci = self.choice_i[0]
            col = self.scores[:,[i]]
            col2 = self.loss_vals[:,[i]]
            idx = np.where(np.isnan(col))[0]
            col[np.isnan(col)] = 0.
            idx = np.append(idx, np.argmax(col))
            # state dict trainied on feats and labels
            model = self.loadmodel(i+1)
            f, l = feats[idx].copy(), labls[idx].copy()
            yield model, f, l, ci, col, col2
            
    def makeaxes(self, n):
        plt.close('all')
        plt.tight_layout()
        fig, ax = plt.subplots(ncols=n, sharex=False, figsize=(11, 5))
        plt.subplots_adjust(wspace=0.3)
        lims = (-1.1, 1.1)
        # add lims to interpolation and scoring axes. in 1d, this will
        # cut off interpolation curve. Do not lim scoring axes
        for i in range(n-1):  
            ax[i].set_xlim(lims)
            ax[i].set_ylim(lims)
        return fig, ax
    
    def sqax(self, ax):
        for i in range(len(ax)):
            x0,x1 = ax[i].get_xlim()
            y0,y1 = ax[i].get_ylim()
            ax[i].set_aspect(abs(x1-x0)/abs(y1-y0))
        return ax
    
    def graph1d(self, x, y, model, choice_i, scores, loss_vals, rnd):
        fig, ax = self.makeaxes(3)
        fig.suptitle(self.title + f"\nRound: {rnd}", y=1.15)
        xt = np.arange(self.data.n)
        ax[0].set_xticks(np.linspace(-1.1, 1.1, self.data.n))
        ax[0].set_xticklabels(xt)
        choice_i = np.argmax(scores)
        ax[0].set_title(f"Training and \n Interpolation \n Choice Idx: {choice_i}")
        ax[1].set_title("Scoring")
        ax[2].set_title("Loss Vals")
        # ax[0]: training set w/ fine interpolation curve
        cbar = ax[0].scatter(x, y, c=y, cmap='bwr', vmin=-1, vmax=1)
        fig.colorbar(cbar, ax=ax[0])
        finex = fine1d()
        ax[0].plot(finex, model.predict(finex))
        # ax[1]: scores
        if scores is not None:
            scores = scores.flatten()
            ax[1].set_ylim(bottom=0, top=np.max(scores[~np.isnan(scores)])*1.1)
            ax[1].bar(xt, scores)
        # ax[2]: loss vals
        if loss_vals is not None:
            loss_vals = loss_vals.flatten()
            ax[2].set_ylim(bottom=0, top=np.max(loss_vals[~np.isnan(loss_vals)])*1.1)
            ax[2].bar(xt, loss_vals)
        for i in range(1, 3):
            ax[i].set_xticks(np.linspace(-1.1, 1.1, self.data.n))
            ax[i].set_xticks(xt)
        ax = self.sqax(ax)
        return fig
    
    def graph2d(self, x, y, model, scores, rnd):
        fig, ax = self.makeaxes(3)
        fig.suptitle(self.title + f"\nRound: {rnd}", y=1.15)
        ticks = np.arange(self.data.n)
        for i in range(2):
            ax[i].set_xticks(np.linspace(-1.1, 1.1, self.data.n))
            ax[i].set_yticks(np.linspace(-1.1, 1.1, self.data.n))
            ax[i].set_xticklabels(ticks)
            ax[i].set_yticklabels(ticks)
        # ax[0]: training set
        ax[0].set_title("Training")
        cbar = ax[0].scatter(x[:,[0]], x[:,[1]], c=y, cmap='bwr', vmin=-1, vmax=1)
        fig.colorbar(cbar, ax=ax[0])
        # ax[1]: interpolation
        ax[1].set_title("Interpolation") 
        finex = fine2d()
        cbar = ax[1].scatter(finex[:,[0]], finex[:,[1]], c=model.predict(finex), cmap='bwr')
        fig.colorbar(cbar, ax=ax[1])
        # ax[2]: scores
        ax[2].set_title("Scoring")
        if scores is not None:
            scores = scores.flatten()
            cbar = ax[2].scatter(self.data.features[:,[0]], self.data.features[:,[1]], c=scores, cmap='magma_r')
            fig.colorbar(cbar, ax=ax[2])
        plt.subplots_adjust(wspace=0.5)
        ax = self.sqax(ax)
        return fig
    
    def graph(self):
        d = self.path + '/graphs/'
        os.makedirs(d, exist_ok=True)
        g = self.graph1d if self.data.dims == 1 else self.graph2d
        for i, subs in enumerate(self.yield_subsets()):
            model, x, y, choice_i, scores, lvals = subs
            fig = g(x, y, model, choice_i, scores, lvals, i)
            fig.savefig(d + str(i).zfill(3) + '.png', bbox_inches="tight")
        return
        