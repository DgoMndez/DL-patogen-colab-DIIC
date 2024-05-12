import pandas as pd
import glob
import ast
import os
import sys
import argparse

# Add src to sys.path to import modules
file_path = os.path.realpath(__file__)
src_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(src_path)

from project_config import *

def get_all_lotes_resultado(batchPath):

    print('Reading batches from ' + batchPath)

    abstractsFiles = glob.glob(batchPath + '/batch*.csv')
    indexFiles = glob.glob(batchPath +'/index*.csv')

    print('Found ' + str(len(abstractsFiles)) + ' abstracts files')
    print('Found ' + str(len(indexFiles)) + ' index files')

    abstractsDfs = []
    indexDfs = []

    for file in abstractsFiles:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        abstractsDfs.append(df)

    for file in indexFiles:
        df = pd.read_csv(file, sep='\t', low_memory=False)
        indexDfs.append(df)

    dfAbstracts = pd.concat(abstractsDfs, ignore_index=True)
    dfIndex = pd.concat(indexDfs, ignore_index=True)

    return dfAbstracts, dfIndex

from pubmed import *

PATH_BATCH_RESULTS = PATH_ABSTRACTS + '/abstracts-shuffled-batches'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Join abstracts batches from a folder into a single file')
    parser.add_argument('-d', '--path', default=PATH_ABSTRACTS, help='Path to abstracts results directory')
    parser.add_argument('-a', '--name', default=ABSTRACTS_NAME, help='Name of the output file')
    parser.add_argument('-i', '--index', default=INDEX_NAME, help='Name of the index file')
    parser.add_argument('-b', '--path_batch', default=PATH_BATCH_RESULTS, help='Path to batches directory')
    args = parser.parse_args()

    path = args.path
    name = args.name.replace('.csv','')
    indexName = args.index.replace('.csv','')
    path_batch = args.path_batch

    results_dir = PATH_ABSTRACTS

    print("Joining abstracts from " + path_batch + " into " + results_dir + '/' + name + '.csv'
          + " and " + results_dir + '/' + indexName + '.csv')
    
    dfAbstracts, dfIndex = get_all_lotes_resultado(path_batch)

    dfAbstracts.to_csv(results_dir + '/' + name + '.csv', sep='\t', index=False)
    dfIndex.to_csv(results_dir + '/' + indexName + '.csv', sep='\t', index=False)

