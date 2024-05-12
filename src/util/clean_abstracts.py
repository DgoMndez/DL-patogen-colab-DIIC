from cmath import nan
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import sys

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from project_config import *

# Función clean abstract

# Download the stopwords from NLTK

nltk.download('punkt')
nltk.download('stopwords')

cached_stopwords = stopwords.words('english')

def clean_abstract(abstract):
    if isinstance(abstract, float) and np.isnan(abstract):
        return ''
    # Convert the text to lowercase
    abstract = abstract.lower() # type: ignore

    # Remove punctuation
    abstract = abstract.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(abstract)

    # Remove stopwords
    tokens = [word for word in tokens if not word in cached_stopwords]

    # Join the tokens back into a single string
    abstract = ' '.join(tokens)

    return abstract

from pubmed import *

OUTPUT_NAME = ABSTRACTS_NAME + '-clean'
OUTPUT_INDEX_NAME = INDEX_NAME + '-clean'

def countPapers(dfAbstracts, phenotypeId):
    return dfAbstracts[dfAbstracts['phenotypeId'] == phenotypeId].shape[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean abstracts from a csv to a new file')
    parser.add_argument('-d', '--path', default=PATH_RESULT, help='Path to abstracts results directory')
    parser.add_argument('-a', '--name', default=ABSTRACTS_NAME, help='Name of the input file')
    parser.add_argument('-i', '--index', default=INDEX_NAME, help='Name of the index file')
    parser.add_argument('-o', '--output', default=OUTPUT_NAME, help='Name of the output abstracts file')
    parser.add_argument('-j', '--output_index', default=OUTPUT_INDEX_NAME, help='Name of the output index file')
    parser.add_argument('-r', '--replace', action='store_true', help='Replace null abstracts with title')
    args = parser.parse_args()

    path = args.path
    name = args.name
    indexName = args.index.replace('.csv','')
    outputName = args.output.replace('.csv','')
    outputIndexName = args.output_index.replace('.csv','')
    replace = args.replace

    abstractsFile = path + '/' + name
    # 1. Load the abstracts
    
    abstracts = pd.read_csv(abstractsFile, sep='\t', low_memory=False, na_values=[''])

    # 2. Clean the abstracts
    
    abstracts['clean_abstract'] = abstracts['abstract'].apply(clean_abstract)
    
    # Substitute empty strings or Nan with title
    if replace:
        abstracts['clean_abstract'] = abstracts['clean_abstract'].mask(abstracts['abstract'].isna(), abstracts['title'].apply(clean_abstract))

    # 3. Remove the original abstracts

    abstracts = abstracts.drop(columns=['abstract'])
    abstracts['length'] = abstracts['clean_abstract'].apply(len)

    # 4. Remove null abstracts

    abstracts = abstracts.dropna(subset=['clean_abstract'])
    abstracts = abstracts.drop(abstracts[abstracts['length'] == 0].index)
    
    # 5. Save the cleaned abstracts to a new file
    abstracts.to_csv(path + '/' + outputName + '.csv', index=False, sep='\t')

    # 6. Get new index to a new file
    # For each phenotype in the index, count the number of papers with that phenotype
    # New dataframe phenotypeId	phenotypeName	numberPapers

    index = pd.read_csv(path + '/' + indexName + '.csv', sep='\t', low_memory=False)

    # Count the number of papers with each phenotype
    index['numberPapers'] = index['phenotypeId'].apply(lambda x: countPapers(abstracts, x))
    index = index.drop(columns=['paperList'])
    
    index.to_csv(path + '/' + outputIndexName + '.csv', index=False, sep='\t')


