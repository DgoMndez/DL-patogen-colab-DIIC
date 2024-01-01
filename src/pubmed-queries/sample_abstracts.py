import collections
import itertools
from Bio import Entrez
import pandas as pd
import os
import logging

# Constants

# Pubmed auth
with open('../../auth/pubmed/email.txt', 'r') as file:
    Entrez.email = file.read().strip()

with open('../../auth/pubmed/api-key.txt', 'r') as file:
    Entrez.api_key = file.read().strip()

PATH_RESULT = "abstracts/"
PATH_FENOTIPOS = "results/phenotypes-22-12-15.csv"
Entrez.sleep_between_tries = 4
SEED = 42
NSAMPLE = 100
RETMAX = 1000

def search(query):
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=RETMAX,
                            retmode='xml',
                            datetype='pdat',
                            mindate='2000/01/01',
                            maxdate='2023/12/31',
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch(ids):
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           rettype='medline',
                           id=ids)
    return Entrez.read(handle)

def get_all_phenotypes():
    df = pd.read_csv(PATH_FENOTIPOS, sep=';', low_memory=False)
    return df

def get_sample_phenotypes(n):
    df = pd.read_csv(PATH_FENOTIPOS, sep=';', low_memory=False)
    return df.sample(n, random_state=SEED)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    # Pasos:
    # 1. Obtener muestra de los fenotipos
    # 2. Para cada fenotipo, obtener todos los ids de los papers
    # 3. Para cada id, obtener el abstract

    # 1. Obtener muestra de los fenotipos
    dfPhen = get_sample_phenotypes(NSAMPLE)

    logging.debug("Muestra de fenotipos: " + str(dfPhen.shape[0]) + "\n")

    # 2. Para cada fenotipo, obtener todos los ids de los papers
    papers_without_abstracts = []
    i = 1
    for index, row in dfPhen.iterrows():
        # Crear subdirectorio para el fenotipo
        name = row['Phenotype']
        idPhen = row['Id']

        logging.debug("Fenotipo " + str(i) + ": " + idPhen + ' - ' + name + "\n")

        dir = PATH_RESULT + '/' + idPhen
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        dfIds = search(name) # parsed XML
        idList = dfIds['IdList']
        
        count = len(idList)

        # Escribir resumen de la búsqueda
        with open(PATH_RESULT + '/' + idPhen + '-summary.txt', 'w') as file:
            file.write('Phenotype: ' + name + '\n')
            file.write('Id: ' + idPhen + '\n')
            file.write('Number of papers: ' + str(count) + '\n')
            file.write('Ids: ' + str(idList) + '\n')

        if int(dfIds["Count"]) == 0:
            logging.debug('No hay papers para ' + idPhen + ' - ' + name + "\n")
            continue

        logging.debug('(' + idPhen + ') ' + str(count) + ' papers:' + str(idList) + '\n')
        
        # 3. Para cada id, obtener los abstracts
        dfAbstracts = fetch(idList)

        j = 1
        for paper in dfAbstracts['PubmedArticle']:
            id = paper['MedlineCitation']['PMID']
            if 'Abstract' in paper['MedlineCitation']['Article']:
                abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            else:
                papers_without_abstracts.append(id)
                logging.debug('Paper ' + str(j) + ' NO TIENE ABSTRACT: ' + str(id) + '\n')
                continue
            logging.debug('Paper ' + str(j) + ' procesado: ' + str(id) + '\n')
            with open(dir + '/' + id + '.txt', 'w') as file:
                file.write(abstract)
            j = j+1
        i = i+1

    logging.debug('Papers sin abstract: ' + str(papers_without_abstracts) + '\n')
    with open(PATH_RESULT + '/papers_without_abstracts.txt', 'w') as file:
        file.write(str(papers_without_abstracts))


