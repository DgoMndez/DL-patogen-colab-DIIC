import collections
import itertools
from Bio import Entrez
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the stopwords from NLTK
#nltk.download('punkt')
#nltk.download('stopwords')

def clean_abstract(abstract):
    # Convert the text to lowercase
    abstract = abstract.lower()

    # Remove punctuation
    abstract = abstract.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(abstract)

    # Remove stopwords
    tokens = [word for word in tokens if not word in stopwords.words()]

    # Join the tokens back into a single string
    abstract = ' '.join(tokens)

    return abstract

PATH_DATA = '../pubmed-queries/abstracts'
PATH_DATA_CSV = PATH_DATA + '/abstracts.csv'
PATH_DATA_FENOTIPOS = '../pubmed-queries/results/phenotypes-22-12-15.csv'
SEED = 42
VERBOSE_COMP = False
if __name__ == '__main__':
    
    #TODO:
    # 1. Imprimir información de los papers del csv: id, titulo y fenotipo
    # 2. Separar en train, validation y test
    # 3. Guardar en un csv los ids de train, validation y test con el abstract

    # 1. Imprimir información de los papers del csv: id, titulo y fenotipo
    if VERBOSE_COMP:
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
        
    dfPapers = pd.read_csv(PATH_DATA_CSV, sep='\t', low_memory=False, na_values=[''])
    dfPhenotypes = pd.read_csv(PATH_DATA_FENOTIPOS, sep=';', low_memory=False, na_values=[''])
    # Assuming dfPapers is your DataFrame
    train_df, test_df = train_test_split(dfPapers, test_size=0.2, random_state=42)

    df = dfPapers.sample(10, random_state=SEED)
    for index, row in df.iterrows():
        abstract = row['abstract']
        abstract = clean_abstract(abstract)
        print(abstract)
        # Fine-tune your BERT transformer with the cleaned abstract


