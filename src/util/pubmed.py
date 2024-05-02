import collections
from datetime import datetime
import itertools
from Bio import Entrez
import pandas as pd
import os
import logging
from urllib.error import HTTPError

# Import global variables from project_config.py

import os
import sys
import time

# Add src to sys.path to import modules
file_path = os.path.realpath(__file__)
src_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(src_path)

from project_config import *

# Pubmed auth
with open(PATH_AUTH+ '/pubmed/email.txt', 'r') as file:
    EMAIL = file.read().strip()

with open(PATH_AUTH + '/pubmed/api-key.txt', 'r') as file:
    PUBMED_API_KEY = file.read().strip()

Entrez.email = EMAIL
Entrez.api_key = PUBMED_API_KEY

PATH_RESULT = PATH_ABSTRACTS
DF_NAME = 'phenotypes_nz_10'
PATH_DFPHEN = PATH_PHENOTYPES + "/" + DF_NAME + ".csv"
PATH_BATCH_DIR = PATH_PHENOTYPES + "/" + DF_NAME + "-batches"
Entrez.sleep_between_tries = 5
SEED = 42
NSAMPLE = 0
RETMAX = 2000
ABSTRACTS_NAME = 'abstracts-'+datetime.today().strftime("%d-%m")
INDEX_NAME = 'index-' + datetime.today().strftime("%d-%m")

ntries = 0
htries = 0

def search(query, retmax=RETMAX):
    try:
        global ntries
        handle = Entrez.esearch(db='pubmed',
                                sort='relevance',
                                retmax=retmax,
                                retmode='xml',
                                datetype='pdat',
                                mindate='2000/01/01',
                                maxdate='2023/12/31',
                                rettype='abstract',
                                term=query)
        results = Entrez.read(handle)
        return results
    except RuntimeError:
        ntries = (ntries + 1) % 180
        logging.debug('Error en search: f{query}. Intento ' + str(ntries) + '\n')
        time.sleep((20*ntries % 3600))
        return search(query, retmax)
    except HTTPError as e:
        htries = (htries + 1)
        logging.error(f'HTTP error: {e}.\n')
        logging.debug('Error en search: f{query}. Intento HTTP ' + str(htries) + '\n')
        time.sleep((20*htries % 3600))
        if htries < 360:
            return search(query, retmax)
        else:
            logging.error('Demasiados intentos. Abortando.\n')
            raise e
    
def fetch(ids):
    global ntries
    try:
        handle = Entrez.efetch(db='pubmed',
                            retmode='xml',
                            rettype='abstract',
                            id=ids)
        results = Entrez.read(handle)
        return results
    except RuntimeError:
        ntries = (ntries + 1) % 180
        logging.debug('Error en fetch. Intento ' + str(ntries) + '\n')
        time.sleep((20*ntries % 3600))
        return fetch(ids)
    except HTTPError as e:
        htries = (htries + 1)
        logging.error(f'HTTP error: {e}.\n')
        logging.debug('Error en fetch: f{query}. Intento HTTP ' + str(htries) + '\n')
        time.sleep((20*htries % 3600))
        if htries < 360:
            return fetch(ids)
        else:
            logging.error('Demasiados intentos. Abortando.\n')
            raise e

def get_all_phenotypes(path=PATH_DFPHEN):
    df = pd.read_csv(path, sep='\t', low_memory=False)
    return df

def get_sample_phenotypes(n, path=PATH_DFPHEN, chunk=0):
    if chunk == 0:
        df = pd.read_csv(path, sep='\t', low_memory=False)
        if n > 0:
            return df.sample(n, random_state=SEED)
    else:
        df = pd.read_csv(path+'/batch-'+str(chunk)+'.csv', sep='\t', low_memory=False)
    return df

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sample abstracts from Pubmed')
    parser.add_argument('--n', default=NSAMPLE, help='Number of phenotypes to sample', type=int)
    parser.add_argument('-d', '--path', default=PATH_RESULT, help='Path to abstracts results directory')
    parser.add_argument('-p', '--phen', default=PATH_DFPHEN, help='Path to phenotypes dataframe')
    parser.add_argument('-r', '--retmax', default=RETMAX, help='Max number of papers to retrieve', type=int)
    parser.add_argument('-a', '--name', default=ABSTRACTS_NAME, help='Name of the output file')
    parser.add_argument('-i', '--index', default=INDEX_NAME, help='Name of the index file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-l', '--lower', default=0, help='Lower bound for batches', type=int)
    parser.add_argument('-u', '--upper', default=1, help='Upper bound for batches', type=int)
    parser.add_argument('-b', '--path_batch', default=PATH_BATCH_DIR, help='Path to batches directory')
    args = parser.parse_args()

    n = args.n
    path = args.path
    phen = args.phen
    retmax = args.retmax
    name = args.name.replace('.csv','')
    indexName = args.index.replace('.csv','')
    verbose = args.verbose
    lower = args.lower
    upper = args.upper
    path_batch = args.path_batch

    lotes = (lower > 0)

    logging.basicConfig(level=logging.DEBUG)
    if not os.path.exists(PATH_RESULT):
        os.makedirs(PATH_RESULT)

    # Pasos:
    # 1. Obtener muestra de los fenotipos
    # 2. Para cada fenotipo, obtener todos los ids de los papers
    # 3. Para cada id, obtener el abstract
    # 4. Escribir información en un csv

    # 1. Obtener muestra de los fenotipos
    
    phenName = phen.split('/')[-1].replace('.csv','')

    if lotes:
        phenPath = path_batch
    else:
        phenPath = phen

    results_dir = PATH_RESULT

    if lotes:
        results_dir = results_dir + '/' + name + '-batches'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    for chunk in range(lower,upper):

        dfPhen = get_sample_phenotypes(n, phenPath, chunk)

        if chunk > 0:
            logging.debug("Lote " + str(chunk) + "\n")
        
        logging.debug("Muestra de fenotipos: " + str(dfPhen.shape[0]) + "\n")

        # 2. Para cada fenotipo, obtener todos los ids de los papers
        papers_without_abstracts = []
        i = 1

        # 4. Escribir información en un csv
        # Header
        if chunk > 0:
            outputCsv = open(results_dir + '/' + 'batch-' + str(chunk) + '.csv', 'w')
            outIndexPhen = open(results_dir + '/' + 'index-' + str(chunk) + '.csv', 'w')
        else:
            outputCsv = open(results_dir + '/' + name + '.csv', 'w')
            outIndexPhen = open(results_dir + '/' + indexName + '.csv', 'w')
        outputCsv.write('paperId\tphenotypeId\tphenotypeName\ttitle\tabstract\n')
        outIndexPhen.write('phenotypeId\tphenotypeName\tnumberPapers\tpaperList\n')

        for index, row in dfPhen.iterrows():
            # Crear subdirectorio para el fenotipo
            name = row['name']
            idPhen = row['id']

            logging.debug("Fenotipo " + str(i) + ": " + idPhen + ' - ' + name + "\n")

            dir = PATH_RESULT + '/text/' + idPhen
            if not os.path.exists(dir):
                os.makedirs(dir)

            query = 'English [Language] ' + name   
            dfIds = search(query, retmax) # parsed XML
            idList = dfIds['IdList']
            
            count = len(idList)

            outIndexPhen.write(idPhen + '\t' + name + '\t' + str(count) + '\t' + ','.join(idList) + '\n')
            if int(dfIds["Count"]) == 0:
                logging.debug('0 papers para ' + idPhen + ' - ' + name + "\n")
                continue

            logging.debug('(' + idPhen + ') ' + str(count) + ' papers')
            if verbose:
                logging.debug(': ' + ','.join(idList) + '\n')
            else:
                logging.debug('\n')
            
            # 3. Para cada id, obtener los abstracts
            dfAbstracts = fetch(idList)

            j = 1
            for paper in dfAbstracts['PubmedArticle']:
                id = paper['MedlineCitation']['PMID']
                if 'Abstract' in paper['MedlineCitation']['Article']:
                    # quizá sea necesario strip('\"')
                    abstract = ''
                    k = 1
                    if 'Abstract' in paper['MedlineCitation']['Article']:
                        for abstractText in paper['MedlineCitation']['Article']['Abstract']['AbstractText']:
                            if 'Label' in abstractText.attributes:
                                text = abstractText.attributes['Label'] + ": " + abstractText
                                if verbose:
                                    logging.debug('Abstract text ' + str(k)
                                            + ' con etiqueta: ' + abstractText.attributes['Label'] + '\n')
                            else:
                                text = abstractText
                                if verbose:
                                    logging.debug('Abstract text ' + str(k)
                                            + ' sin etiqueta: ' + str(id) + '\n')
                            if k > 1:
                                abstract = abstract + ' ' + text
                            else:
                                abstract = abstract + text
                            k = k+1
                    if verbose:
                        logging.debug('Paper ' + str(j) + ' procesado: ' + str(id) + '\n')
                        logging.debug('Abstract: ' + abstract + '\n')
                    abstract = abstract.strip('"') # Duda existencial
                    with open(dir + f'/{idPhen}-all.txt', 'a') as file:
                        file.write(id + '\n')
                        file.write(abstract)
                        file.write('\n\n')
                else:
                    papers_without_abstracts.append(id)
                    logging.debug('Paper ' + str(j) + ' NO TIENE ABSTRACT: ' + str(id) + '\n')
                    abstract=''
                # 4. Escribir información en un csv
                outputCsv.write(id + '\t' + idPhen + '\t' + name + '\t'
                    + "\"" + paper['MedlineCitation']['Article']['ArticleTitle'] + '\"\t'
                    + "\"" + abstract + '\"\n')
                j = j+1
            i = i+1

        papers_string = ','.join(papers_without_abstracts)
        logging.debug('Papers sin abstract: ' + papers_string + '\n')
        with open(results_dir + '/papers_without_abstracts.txt', 'w') as file:
            count = len(papers_without_abstracts)
            file.write(str(count)+'\n')
            file.write(papers_string+'\n')
