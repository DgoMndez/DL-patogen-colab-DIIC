import collections
import itertools
from Bio import Entrez
import pandas as pd
import os
import logging

PATH_DATA = '../pubmed-queries/abstracts'
PATH_DATA_CSV = PATH_DATA + '/abstracts.csv'
if __name__ == '__main__':
    
    #TODO:
    # 1. Imprimir información de los papers del csv: id, titulo y fenotipo
    # 2. Separar en train, validation y test
    # 3. Guardar en un csv los ids de train, validation y test con el abstract

    # 1. Imprimir información de los papers del csv: id, titulo y fenotipo
    for cad in [PATH_DATA_CSV, PATH_DATA+'/index-phenotypes.csv']:
        df = pd.read_csv(cad, sep='\t', low_memory=False, na_values=[''])
        print(df.shape)
        j = 1
        for index, row in df.iterrows():
            l = []
            for col in df.columns:
                l.append(str(row[col]))
            print(';'.join(l))
            j = j+1
            if j >= 1000:
                break

