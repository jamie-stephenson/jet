from src.dist_utils import split_txt

def get_dataset(path,rank,world_size):
    path+='test.txt'
    return {'text':split_txt(path,rank,world_size).decode(errors='replace')}