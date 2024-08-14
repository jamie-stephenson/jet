import importlib

def get_dataset(name, path, rank, world_size):
    
    dataset_module = importlib.import_module("project_datasets." + name, package=".")
    dataset = dataset_module.get_dataset(path,rank,world_size)

    return dataset

def download_dataset(name,path,nproc):
        
    dataset_module = importlib.import_module("project_datasets." + name, package=".")
    dataset_module.download_dataset(path,nproc)