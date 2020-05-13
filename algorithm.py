import numpy as np
import time
import torch
import os
import json
from dataset import load_dataset, load_dataset_path
from model import model
from distributed import worker_client
from itertools import product
from collections.abc import Iterable

class algorithm:
    
    def __init__(self, dataset, model_builder):
        ''' class to run algorithm
        @params:
            dataset: instance of ALNet3.dataset
            model_builder: function which returns ALNet3. model instance
            '''
        self.data = dataset
        self.data.init_training_mask()
        self.model_builder = model_builder
        self.r = 0
        self.log_ = {
            'state_dicts': {}, # state dicts for each round
            'scores': []   # list of [[(i1, i2), (i2, s2)], ...]
        }
        
    def build_model(self):
        return self.model_builder()
    
    def log(self, sd, scores):
        self.log_['state_dicts'][str(self.r)] = sd
        self.r += 1
        if scores is not None:
            self.log_['scores'].append(scores)
        return len(self.log_['state_dicts'])
    
    def start(self):
        print("Start")
        m = self.build_model()
        x, y = self.data.labeled_points
        m.learn(x, y)
        idxs = self.data.labeled_idxs
        sd = m.state_dict()
        del m
        return self.log(sd, None)
    
    def explore_label(self, x, y, label):
        y[[-1]] = label
        m = self.build_model()
        _, score, _ = m.learn(x, y)
        del m
        return score, label
    
    def explore_labels(self, x, y):
        labels = [[1.], [-1.]]
        futures = []
        with worker_client() as wc:
            for label in labels:
                futures.append(wc.submit(self.explore_label, x, y, label))
            futures = wc.gather(futures)
        scores = [f[0] for f in futures]
        i = np.argmin(scores)
        return scores[i], labels[i]

    def explore_round(self):
        widx, wscr = -1, -1
        scores = []
        idxs = []
        futures = []
        with worker_client() as wc:
            for idx, feats, labls in self.data.yield_unlabeled_points():
                futures.append(wc.submit(self.explore_labels, feats, labls))
                idxs.append(idx)
            futures = wc.gather(futures)
        scores = [f[0] for f in futures]
        i = np.argmax(scores)
        return idxs[i], list(zip(idxs, scores))
    
    def explore(self):
        while self.data.has_unlabeled:
            print("Round: ", self.r)
            choice_i, scores = self.explore_round()
            self.data.mark_labeled(choice_i)
            x, y = self.data.labeled_points
            m = self.build_model()
            loss, score, state_dict = m.learn(x, y)
            self.log(state_dict, scores)
            time.sleep(0.25)
        return

    def format_results(self):
        res = self.log_
        scores = res['scores']
        mat = np.full((len(self.data.labels), len(scores)), np.nan)
        for i, s in enumerate(scores):
            mask = np.array([False] * len(self.data.labels))
            s = np.array(s)
            mask[s[:,0].astype(int)] = True
            mat[:,i][mask] = s[:,1]
        self.log_['scores'] = mat
        return
    
    def run(self):
        self.start()
        self.explore()
        self.format_results()
        return self.log_

class saver:
    
    def __init__(self):
        self.pref = './Results/'
        
    def json_algo(self, datapath, model_args, crit, optim, optim_args, epochs):
        js = {
            'data': datapath, 
            'criterion': str(crit).split('.')[-1][:-2],
            'optimizer': str(optim).split('.')[-1][:-2],
            'epochs': str(epochs),
        }
        js['model_args'] = list(map(lambda x: str(x), model_args))
        k = list(optim_args)
        v = [str(optim_args[key]) for key in k]
        js['optim_args'] = [k, v]
        return js
        
    def save(self, name, algo, model_args, crit, optim, optim_args, epochs):
        os.makedirs(self.pref, exist_ok=True)
        path = self.pref + name
        os.makedirs(path, exist_ok=True)
        pt_path = path + '/state_dicts.pt'
        torch.save(algo.log_['state_dicts'], pt_path)
        np.save(path + '/scores.npy', algo.log_['scores'])
        with open(path + '/algo.json', 'w') as fp:
            fp.write(
                json.dumps(self.json_algo(algo.data.path, model_args, 
                  crit, optim, optim_args, epochs), indent=4)
            )
        return path
    
    
class loader:
    
    def __init__(self, name):
        self.path = name + '/'
        
    def load(self):
        ''' get data, model builder, state_dicts, and results '''
        with open(self.path + 'algo.json', 'r') as fp:
            js = json.loads(fp.read())
        data = load_dataset_path(js['data'])
        pt = torch.load(self.path + 'state_dicts.pt')
        scores = np.load(self.path + 'scores.npy')
        model_args = [int(_) for _ in js['model_args']]
        optim_args = {}
        for i, oa in enumerate(js['optim_args'][0]):
            optim_args[oa] = float(js['optim_args'][1][i])
        crit = getattr(torch.nn, js['criterion'])
        optim = getattr(torch.optim, js['optimizer'])
        epochs = int(js['epochs'])
        
        def mb():
            m = model(*model_args, crit(), optim, optim_args, epochs)
            return m
        
        return data, pt, scores, model_args, crit, optim, optim_args, mb
            
        
class Experiment:
    
    def __init__(self, name, pset={}):
        self.name = name
        self.pset = pset
        
    def iterpsets(self):
        keys = list(self.pset)
        vals = []
        for key in keys:
            if not isinstance(self.pset[key], Iterable):
                self.pset[key] = [self.pset[key]]
            vals.append(self.pset[key])
        for i, v in enumerate(product(*vals)):
            foo = {keys[i]: v[i] for i in range(len(v))}
            yield i, foo.copy()
    
    def go_(self, **kw):
        d = load_dataset(kw.get('dims', 1), kw.get('n', 10))

        crit = kw.get('criterion', torch.nn.MSELoss)
        optim = kw.get('optimizer', torch.optim.Adam)
        optim_args = kw.get('optim_args', {"lr": 1e-3, "weight_decay": 1e-5})
        model_args = kw.get('model_args', 
            (kw.get('dims', 1), kw.get('model_Nodes', 10), kw.get('model_dimOut', 1))
        )
        epochs = kw.get('epochs', 1)

        def mb():
            m = model(*model_args, crit(), optim, optim_args, epochs)
            return m

        # run algo
        algo = algorithm(d, mb)
        res = algo.run()

        # save algo
        s = saver()
        return s.save(kw.get('name', self.name + '/' + str(kw.get('pset_i', 0))), \
                      algo, model_args, crit, optim, optim_args, epochs)

    def go(self):
        with worker_client() as wc:        
            futures = [wc.submit(self.go_, pset_i=i, **pset) \
                       for i, pset in self.iterpsets()]
            futures = wc.gather(futures)
        return futures