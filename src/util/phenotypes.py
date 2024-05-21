# Import global variables from project_config.py

from datetime import datetime
import os
import sys

import numpy as np
import pandas as pd
from pyhpo import Ontology
from queue import Queue

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from project_config import *

onto = Ontology(PATH_ONTO)

def getSubOntologyDf(phenName, filename=None, ic_kind='gene'):
    nodo = onto.get_hpo_object(phenName)
    sAux = Queue()
    sAux.put([nodo,0])
    visitados = {}
    phenAttributes = ['id', 'name', 'def', 'depth', 'isLeaf', 'ic', 'ic_gene', 'ic_omim']
    phenValues = []
    while not sAux.empty():
        nodo, depth = sAux.get()
        if nodo in visitados:
            continue
        visitados[nodo] = 1
        phen = nodo
        isLeaf = False
        if phen.children:
            for hijo in phen.children:
                sAux.put([hijo, depth+1])
        else: isLeaf = True
        ic = phen.information_content
        phenValues.append([phen.id, phen.name, phen.definition, depth, isLeaf, ic[ic_kind], ic.gene, ic.omim])
    dfPhen = pd.DataFrame(phenValues, columns=phenAttributes, index=None)
    if filename:
        dfPhen.to_csv(filename, index=False, sep='\t')
    return dfPhen

def getSelection(dfPhen, depth, filename=None):
    df = dfPhen[dfPhen['ic'] > 0]
    dfLess = df[(df['depth'] < depth) & df['isLeaf']]
    dfEq = df[df['depth'] == depth]
    dfTrue = pd.concat([dfLess, dfEq])
    if filename:
        dfTrue.to_csv(filename, index=False, sep='\t')
    return dfTrue

def getNull(dfPhen, filename=None):
    df = dfPhen[(dfPhen['ic_gene'] == 0) | (dfPhen['ic_omim'] == 0)]
    if filename:
        df.to_csv(filename, index=False, sep='\t')
    return df

# https://stackoverflow.com/a/28882020/23198260
def split_dataframe(df, num_chunks=10): 
    chunks = list()
    chunk_size = len(df) // num_chunks
    m = len(df) % num_chunks
    prev = 0
    for i in range(m):
        chunks.append(df[prev:prev+chunk_size+1])
        prev += chunk_size + 1
    for i in range(m, num_chunks):
        chunks.append(df[prev:prev+chunk_size])
        prev += chunk_size
    return chunks

def csv2Lotes(filename, n, path, name):
    df = pd.read_csv(filename, sep='\t')
    writeLotes(df, n, path, name)

def writeLotes(df, n, path, name):
    df_shuffled = df.sample(frac=1)
    df_split = split_dataframe(df_shuffled, n)
    i = 1
    dir = path + '/' + name + '-batches/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for df_sample in df_split:
        df_sample.to_csv(dir+'batch-' + str(i) + '.csv', sep='\t', index=False)
        i += 1

def getPhenotypeName(phenotypeId):
    return onto.get_hpo_object(phenotypeId).name

PATH_RESULTS = PATH_PHENOTYPES
NAME_DFPHEN = 'phenotypic_abnormality'
NAME_SELECT = 'phenotypes_nz_' + datetime.today().strftime("%d-%m")

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get phenotype csv files from HPO:PA')
    parser.add_argument('--phen', default=ROOTPHEN, help='Root phenotype')
    parser.add_argument('-d', '--depth', default=10, help='Depth selection', type=int)
    parser.add_argument('-p','--path', default=PATH_RESULTS, help='Path to results directory')
    parser.add_argument('-a','--name_all', default=NAME_DFPHEN, help='Filename for all phenotypes csv')
    parser.add_argument('-s','--name_select', default=NAME_SELECT, help='Filename for selection csv')
    parser.add_argument('-b', '--batches', default=0, help='Number of batches', type=int)
    parser.add_argument('-l', '--blength', default=0, help='Batch length', type=int)
    parser.add_argument('-w', '--write', action='store_true', help='Just divide the file in batches')

    args = parser.parse_args()

    phen = args.phen
    depth = args.depth
    path_all = args.path
    name_all = args.name_all.replace('.csv','')
    name_select = args.name_select.replace('.csv','')
    batches = args.batches
    l = args.blength
    w = args.write

    if w:
        dfTrue = pd.read_csv(path_all + '/' + name_select + f'_{depth}'+'.csv', sep='\t')
    else:
        dfPhen = getSubOntologyDf(phen, path_all + '/' + name_all + '.csv')
        dfTrue = getSelection(dfPhen, depth, path_all + '/' + name_select + f'_{depth}'+'.csv')

    if l > 0:
        writeLotes(dfTrue, int(len(dfTrue)/l), path_all, name_select + f'_{depth}')
    elif batches > 0:
        writeLotes(dfTrue, batches, path_all, name_select + f'_{depth}')

