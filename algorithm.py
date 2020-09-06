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
            'choice_i': [],    # list of [wi1, wi2, ...]
            'scores': [],      # list of [[(i1, s1), (i2, s2)], ...]
            'loss_vals': []    # list of [[(i1, l1), (i2, l2)], ...]
        }
        
    def build_model(self):
        return self.model_builder()
    
    def log(self, sd, choice_i, scores, loss_vals):
        self.log_['state_dicts'][str(self.r)] = sd
        self.r += 1
        if choice_i is not None:
            self.log_['choice_i'].append(choice_i.copy())
        if scores is not None:
            self.log_['scores'].append(scores.copy())
        if loss_vals is not None:
            self.log_['loss_vals'].append(loss_vals.copy())
        return len(self.log_['state_dicts'])
    
    def start(self):
        print("Start")
        m = self.build_model()
        x, y = self.data.labeled_points
        m.learn(x, y)
        idxs = self.data.labeled_idxs
        sd = m.state_dict()
        del m
        return self.log(sd, None, None, None)
    
    def explore_label(self, x, y, label):
        y[[-1]] = label
        m = self.build_model()
        loss_val, score, _ = m.learn(x, y)
        del m
        return loss_val, score, label
    
    def explore_labels(self, x, y):
        labels = [[1.], [-1.]]
        futures = []
        with worker_client() as wc:
            for label in labels:
                futures.append(wc.submit(self.explore_label, x, y, label))
            futures = wc.gather(futures)
        loss_vals = [f[0] for f in futures]
        scores = [f[1] for f in futures]
        i = np.argmin(scores)
        return np.float64(loss_vals[i]), scores[i], labels[i]

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
        loss_vals = [f[0] for f in futures]
        scores = [f[1] for f in futures]
        i = np.argmax(scores)
        i_scores = list(zip(idxs.copy(), scores))
        i_lvals = list(zip(idxs.copy(), loss_vals))
        return idxs[i], i_scores, i_lvals
    
    def explore(self):
        while self.data.has_unlabeled:
            print("Round: ", self.r)
            choice_i, scores, loss_vals = self.explore_round()
            self.data.mark_labeled(choice_i)
            x, y = self.data.labeled_points
            m = self.build_model()
            loss, score, state_dict = m.learn(x, y)
            self.log(state_dict, choice_i, scores, loss_vals)
            time.sleep(0.25)
        return
    
    def _format_result(self, k):
        res = self.log_
        key = res[k]
        mat = np.full((len(self.data.labels), len(key)), np.nan)
        for i, s in enumerate(key):
            mask = np.array([False] * len(self.data.labels))
            s = np.array(s)
            mask[s[:,0].astype(int)] = True
            mat[:,i][mask] = s[:,1]
        self.log_[k] =  mat
        return

    def format_results(self):
        self._format_result('scores')
        self._format_result('loss_vals')
        return
    
    def run(self):
        self.start()
        self.explore()
        print("done exploring!")
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
        np.save(path + '/choice_i.npy', algo.log_['choice_i'])
        np.save(path + '/scores.npy', algo.log_['scores'])
        np.save(path + '/loss_vals.npy', algo.log_['loss_vals'])
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
        choice_i = np.load(self.path + 'choice_i.npy')
        scores = np.load(self.path + 'scores.npy')
        loss_vals = np.load(self.path + 'loss_vals.npy')
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
        loaded = {
            'data': data, 'pt': pt, 'scores': scores, 'loss_vals': loss_vals,
            'choice_i': choice_i, 'crit': crit, 'optim': optim, 
            'optim_args': optim_args, 'mb': mb
        }
        return loaded
            
        
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
        d = load_dataset(kw.get('dims', 1), kw.get('n', 10), spline=kw.get('spline', 5))
        crit = kw.get('criterion', torch.nn.MSELoss)
        optim = kw.get('optimizer', torch.optim.Adam)
        optim_args = kw.get('optim_args', {
            "lr": kw.get('optim_lr', 1e-3), 
            "weight_decay": kw.get('optim_weight_decay', 1e-5),
            }
        )
        model_args = kw.get('model_args', 
            (kw.get('dims', 1), kw.get('model_Nodes', 10), kw.get('model_dimOut', 1))
        )
        epochs = kw.get('epochs', 1)

        def mb():
            m = model(*model_args, crit(), optim, optim_args, epochs)
            return m

        # run algo
        algo = algorithm(d, mb)
        print(algo.data.path)
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