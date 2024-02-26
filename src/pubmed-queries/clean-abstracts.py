from cmath import nan
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Función clean abstract

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

PATH_DATA = './abstracts'
PATH_DATA_CSV = PATH_DATA + '/abstracts.csv'

cached_stopwords = stopwords.words('english')

def clean_abstract(abstract):
    if isinstance(abstract, float) and np.isnan(abstract):
        return ''
    # Convert the text to lowercase
    abstract = abstract.lower()

    # Remove punctuation
    abstract = abstract.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(abstract)

    # Remove stopwords
    tokens = [word for word in tokens if not word in cached_stopwords]

    # Join the tokens back into a single string
    abstract = ' '.join(tokens)

    return abstract

# 1. Load the abstracts
abstracts = pd.read_csv(PATH_DATA_CSV, sep='\t', low_memory=False, na_values=['', nan])
# 2. Clean the abstracts
abstracts['clean_abstract'] = abstracts['abstract'].apply(clean_abstract)
# Substitute empty strings or Nan with title

abstracts['clean_abstract'] = abstracts['clean_abstract'].mask(abstracts['abstract'].isna(), abstracts['title'].apply(clean_abstract))

# 3. Remove the original abstracts
abstracts = abstracts.drop(columns=['abstract'])

# 4. Save the cleaned abstracts to a new file
abstracts.to_csv(PATH_DATA + '/abstracts-clean.csv', index=False, sep='\t')