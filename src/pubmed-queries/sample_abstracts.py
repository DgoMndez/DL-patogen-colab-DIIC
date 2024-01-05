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

PATH_RESULT = "abstracts"
PATH_FENOTIPOS = "results/phenotypes-22-12-15.csv"
Entrez.sleep_between_tries = 5
SEED = 42
NSAMPLE = 100
RETMAX = 500

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
    return results

def fetch(ids):
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           rettype='abstract',
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
    if not os.path.exists(PATH_RESULT):
        os.makedirs(PATH_RESULT)
    # Pasos:
    # 1. Obtener muestra de los fenotipos
    # 2. Para cada fenotipo, obtener todos los ids de los papers
    # 3. Para cada id, obtener el abstract
    # 4. Escribir información en un csv

    # 1. Obtener muestra de los fenotipos
    dfPhen = get_sample_phenotypes(NSAMPLE)

    logging.debug("Muestra de fenotipos: " + str(dfPhen.shape[0]) + "\n")

    # 2. Para cada fenotipo, obtener todos los ids de los papers
    papers_without_abstracts = []
    i = 1

    # 4. Escribir información en un csv
    # Header
    outputCsv = open(PATH_RESULT + '/abstracts.csv', 'w')
    outputCsv.write('paperId\tphenotypeId\tphenotypeName\ttitle\tabstract\n')
    outIndexPhen = open(PATH_RESULT + '/index-phenotypes.csv', 'w')
    outIndexPhen.write('phenotypeId\tphenotypeName\tnumberPapers\tpaperList\n')

    for index, row in dfPhen.iterrows():
        # Crear subdirectorio para el fenotipo
        name = row['Phenotype']
        idPhen = row['Id']

        logging.debug("Fenotipo " + str(i) + ": " + idPhen + ' - ' + name + "\n")

        dir = PATH_RESULT + '/text/' + idPhen
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        dfIds = search(idPhen) # parsed XML
        idList = dfIds['IdList']
        
        count = len(idList)

        outIndexPhen.write(idPhen + '\t' + name + '\t' + str(count) + '\t' + ','.join(idList) + '\n')
        if int(dfIds["Count"]) == 0:
            logging.debug('0 papers para ' + idPhen + ' - ' + name + "\n")
            continue

        logging.debug('(' + idPhen + ') ' + str(count) + ' papers:' + str(idList) + '\n')
        
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
                            logging.debug('Abstract text ' + str(k)
                                          + ' con etiqueta: ' + abstractText.attributes['Label'] + '\n')
                        else:
                            text = abstractText
                            logging.debug('Abstract text ' + str(k)
                                          + ' sin etiqueta: ' + str(id) + '\n')
                        if k > 1:
                            abstract = abstract + ' ' + text
                        else:
                            abstract = abstract + text
                        k = k+1
                logging.debug('Paper ' + str(j) + ' procesado: ' + str(id) + '\n')
                logging.debug('Abstract: ' + abstract + '\n')
                abstract = abstract.strip('"') # Duda existencial
                with open(dir + '/' + id + '.txt', 'w') as file:
                    file.write(abstract)
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
    with open(PATH_RESULT + '/papers_without_abstracts.txt', 'w') as file:
        count = len(papers_without_abstracts)
        file.write(str(count)+'\n')
        file.write(papers_string+'\n')