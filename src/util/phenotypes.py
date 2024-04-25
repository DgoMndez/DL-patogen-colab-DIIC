# Import global variables from project_config.py

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

PATH_RESULTS = PATH_PHENOTYPES
NAME_DFPHEN = 'phenotypic_abnormality'
NAME_SELECT = 'phenotypes_nz'

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get phenotype csv files from HPO:PA')
    parser.add_argument('--phen', default=ROOTPHEN, help='Root phenotype')
    parser.add_argument('--depth', default=10, help='Depth selection', type=int)
    parser.add_argument('-p','--path', default=PATH_RESULTS, help='Path to results directory')
    parser.add_argument('-a','--name_all', default=NAME_DFPHEN, help='Filename for all phenotypes csv')
    parser.add_argument('-s','--name_select', default=NAME_SELECT, help='Filename for selection csv')

    args = parser.parse_args()

    phen = args.phen
    depth = args.depth
    path_all = args.path
    name_all = args.name_all.replace('.csv','')
    name_select = args.name_select.replace('.csv','')

    dfPhen = getSubOntologyDf(phen, path_all + '/' + name_all + '.csv')
    dfTrue = getSelection(dfPhen, depth, path_all + '/' + name_select + f'_{depth}'+'.csv')