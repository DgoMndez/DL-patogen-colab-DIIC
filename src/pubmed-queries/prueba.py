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
RETMAX = 1000

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

if __name__ == '__main__':
    handle = fetch([36396193])
    abstract = handle['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
    print(abstract) # error: tiene un " al final