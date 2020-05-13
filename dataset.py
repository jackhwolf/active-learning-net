import numpy as np
import os
import matplotlib.pyplot as plt

SPLINE = 5
DECISION = 'linear'
CMAP = 'bwr'

class dataset:
    ''' base class for a dataset. will handle tracking which points
    are labeled and size/shape of dataset '''
    
    def __init__(self, dims, n):
        self.features = np.zeros((n, dims), dtype=np.float32) 
        self.labels = np.zeros((n,1), dtype=np.float32)
        self.dims = dims
        self.n = n
        self.desc = None
        self.pref = './data'
        self.fine_data_path = self.pref + '/' + f"fine-{dims}.npy"
        self.labeled_mask = None
        
    @property
    def path(self):
        return self.pref + '/' + self.desc
    
    def save(self):
        os.makedirs(self.pref, exist_ok=True)
        x = np.hstack((self.features, self.labels)).astype(np.float32)
        np.save(self.path + '.npy', x)
        return self.path
    
    def init_training_mask(self):
        x = self.features
        mask = np.array([False] * x.shape[0])
        if x.shape[-1] == 1:
            mask[np.where(x.flatten() == -1)[0]] = True
            mask[np.where(x.flatten() ==  1)[0]] = True
        elif x.shape[-1] == 2:
            # i1 := x == median(x) and y == max(y)
            # i2 := x == median(x) and y == min(y)
            _ = x[:,0] == 0
            i1 = np.where((_) & (x[:,1] == 1))[0]
            i2 = np.where((_)  & (x[:,1] == -1))[0]
            mask[i1] = True
            mask[i2] = True
        else:
            pass
        self.labeled_mask = mask
        return 
    
    @property
    def labeled_idxs(self):
        return np.where(self.labeled_mask)[0]
    
    @property
    def unlabeled_idxs(self):
        return np.where(~self.labeled_mask)[0]
    
    @property
    def has_unlabeled(self):
        return np.any(self.unlabeled_idxs)
    
    @property
    def labeled_points(self):
        return self.features[self.labeled_idxs], self.labels[self.labeled_idxs]
        
    def yield_unlabeled_points(self):
        idxs = self.unlabeled_idxs
        feats, labls = self.labeled_points
        feats = np.append(feats, feats[[-1]], axis=0)
        labls = np.append(labls, labls[[-1]], axis=0)
        for i in idxs:
            feats[-1] = self.features[i]
            labls[-1] = self.labels[i]
            yield i, feats.copy(), labls.copy()
            
    def mark_labeled(self, idx):
        self.labeled_mask[idx] = True
        return idx

class dataset1D(dataset):
    
    def __init__(self, n, spline=SPLINE, **kw):
        spline = int(spline)
        super(dataset1D, self).__init__(1, n)
        self.features = np.linspace(-1, 1, self.n).reshape(self.n, 1)
        label = -1
        for i in range(self.n):
            if i % spline == 0:
                label *= -1
            self.labels[i] = label
        self.desc = f"dims=1&n={n}&spline={spline}"
        
    def graph(self):
        fig, ax = plt.subplots()
        cbar = ax.scatter(self.features, self.labels, c=self.labels, cmap='bwr', vmin=-1, vmax=1)
        fig.colorbar(cbar)
        path = self.path + '.png'
        fig.savefig(path)
        return path
    
# for making 2D datasets
class dataset2D(dataset):
    
    def __init__(self, n, decision=DECISION, **kw):
        super(dataset2D, self).__init__(2, n)
        self.desc = f"dims=2&n={n}&decision={decision}"
        self.decision = getattr(self, decision)
        data = []
        xpts = np.linspace(-1, 1, self.n)
        ypts = sorted(np.linspace(-1, 1, self.n), reverse=True)
        for xpt in xpts:
            for ypt in ypts:
                data.append([xpt, ypt, self.decision(xpt, ypt)])
        self.features = np.array(data).astype(np.float32)
        self.features, self.labels = self.features[:,:-1], self.features[:,[-1]]
                
    def diagonal(self, x, y):
        return 1 if x > y else -1
    
    def linear(self, x, y):
        return 1 if y > 0 else -1

    def graph(self):
        fig, ax = plt.subplots()
        cbar = ax.scatter(self.features[:,[0]], self.features[:,[1]], c=self.labels, cmap=CMAP, vmin=-1, vmax=1)
        fig.colorbar(cbar)
        path = self.path + '.png'
        fig.savefig(path)
        return path
    
    
######################
##      util        ##
######################

def load_dataset_path(path):
    path += '.npy'
    x =  np.load(path)
    path = path.split('/')[-1].split('.')[0].split('&')
    kws = {}
    names = ['dims', 'n', 'decision', 'spline']
    for n in names:
        for p in path:
            p = p.split('=')
            if p[0] == n:
                kws[n] = p[1]
    kw = kws.copy()
    del kw['dims']
    del kw['n']
    return construct_dataset(x, int(kws['dims']), int(kws['n']), **kw)

def load_dataset(dims, n, fine=False, **kw):
    path = './data/dims={}&n={}.npy'
    if dims == 1:
        path = path.format(dims, f"{n}&spline={kw.get('spline', SPLINE)}")
    else:
        path = path.format(dims, f"{n}&decision={kw.get('decision', 'linear')}")
    return construct_dataset(np.load(path), dims, n)

def construct_dataset(mat, dims, n, **kw):
    if dims == 1:
        d = dataset1D(n, **kw)
    else:
        d = dataset2D(n, **kw)
    x, y = mat[:,:-1], mat[:,[-1]]
    d.features = x
    d.labels = y
    return d
            
def make_all_datasets():
    for i in [2, 4, 6]:
        d = dataset1D(i*SPLINE)
        d.save()
        print(d.graph())
    for i in (5, 11, 15, 21):
        for dec in ['linear', 'diagonal']:
            d = dataset2D(i, decision=dec)
            d.save()
            print(d.graph())

        

