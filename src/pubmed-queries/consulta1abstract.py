import collections
import itertools
from Bio import Entrez
import pandas as pd
import os
import logging
from io import StringIO
from Bio import Entrez, Medline

# Constants

# Pubmed auth
with open('../../auth/pubmed/email.txt', 'r') as file:
    Entrez.email = file.read().strip()

with open('../../auth/pubmed/api-key.txt', 'r') as file:
    Entrez.api_key = file.read().strip()

PATH_RESULT = "results/abstracts"
PATH_FENOTIPOS = "results/phenotypes-22-12-15.csv"
Entrez.sleep_between_tries = 5
SEED = 42
NSAMPLE = 1
RETMAX = 5

def search(query):
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=RETMAX,
                            retmode='xml',
                            datetype='pdat',
                            mindate='2000/01/01',
                            maxdate='2023/12/31',
                            rettype='abstract',
                            term=query)
    results = Entrez.read(handle)
    handle.close()
    return results

def fetch(ids):
    handle = Entrez.efetch(db='pubmed',
                           retmode='text',
                           rettype='medline',
                           id=ids)
    return handle

def get_all_phenotypes():
    df = pd.read_csv(PATH_FENOTIPOS, sep=';', low_memory=False)
    return df

def get_sample_phenotypes(n):
    df = pd.read_csv(PATH_FENOTIPOS, sep=';', low_memory=False)
    return df.sample(n, random_state=SEED)

def search_medline(query):
    search = Entrez.esearch(db='pubmed', term=query, usehistory='y',
                            sort='relevance',
                            retmax=RETMAX,
                            datetype='pdat',
                            mindate='2000/01/01',
                            maxdate='2023/12/31',
                            rettype='abstract')
    handle = Entrez.read(search)
    try:
        return handle
    except Exception as e:
        raise IOError(str(e))
    finally:
        search.close()

def fetch_rec(rec_id, entrez_handle):
    fetch_handle = Entrez.efetch(db='pubmed', id=rec_id,
                                 rettype='Medline', retmode='text',
                                 webenv=entrez_handle['WebEnv'],
                                 query_key=entrez_handle['QueryKey'])
    rec = fetch_handle.read()
    return rec

def main(query, email):
    rec_handler = search_medline(query, email)

    for rec_id in rec_handler['IdList']:
        rec = fetch_rec(rec_id, rec_handler)
        rec_file = StringIO(rec)
        medline_rec = Medline.read(rec_file)
        if 'AB' in medline_rec:
            print(medline_rec['AB'])

if __name__ == '__main__':

    # Pasos:
    # 1. Obtener 1 fenotipo
    # 2. Obtener todos los ids de los papers
    # 3. Para cada id, obtener el abstract
    # 4. Escribir el abstract

    # 1. Obtener el fenotipo
    dfPhen = get_sample_phenotypes(NSAMPLE)
    name = dfPhen['Phenotype']
    idPhen = dfPhen['Id']

    # 2. Obtener todos los ids de los papers

    logging.debug("Fenotipo : " + idPhen + ' - ' + name + "\n")

    dir = PATH_RESULT + '/text/' + str(idPhen)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    dfIds = search(name) # parsed XML
    idList = dfIds['IdList']
    
    count = len(idList)

    print(idPhen + '\t' + name + '\t' + str(count) + '\t' + ','.join(idList) + '\n')
    if int(dfIds["Count"]) == 0:
        logging.debug('0 papers para ' + idPhen + ' - ' + name + "\n")
        exit(0)

    logging.debug('(' + idPhen + ') ' + str(count) + ' papers:' + str(idList) + '\n')
    
    # 3. Para cada id, obtener los abstracts
    


    rec_handler = search_medline(name)
    for rec_id in rec_handler['IdList']:
        rec = fetch_rec(rec_id, rec_handler)
        rec_file = StringIO(rec)
        medline_rec = Medline.read(rec_file)
        if 'AB' in medline_rec:
            print("Abstract for: " + str(rec_id))
            print(medline_rec['AB'])
