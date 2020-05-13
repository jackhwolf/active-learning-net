'''
entry point for running algorithm distributed thru dask
'''

if __name__ == '__main__':
    from distributed import Client
    
    cli = Client('144.92.142.184:8786')
    for f in ['model', 'dataset', 'script', 'algorithm', 'visualizer']:
        cli.upload_file(f + '.py')

    from algorithm import Experiment
    from visualizer import Visualizer
    
    pset = {
        'dims': 2,
        'n': 5,
        'epochs': 100,
    }

    
    e = Experiment("May13-2D-test", pset=pset)
    
    futures = cli.submit(e.go)
    futures = cli.gather(futures)
    
    for f in futures:
        v = Visualizer(f)
        v.graph()
        print(f)
    
    print("Done")
    
