import os

# Project structure
PATH_CONFIG = os.path.realpath(__file__)
PATH_SRC = os.path.dirname(os.path.dirname(PATH_CONFIG))
PATH_PROJECT = os.path.dirname(PATH_SRC)
PATH_DATA = os.path.join(PATH_PROJECT, 'data')
PATH_OUTPUT = os.path.join(PATH_PROJECT, 'output')

# Required to update with your info
PATH_AUTH = os.path.join(PATH_PROJECT, 'auth')

# Useful data dirs
PATH_ABSTRACTS = os.path.join(PATH_DATA, 'abstracts')
PATH_PHENOTYPES = os.path.join(PATH_DATA, 'phenotypes')
PATH_ONTO = os.path.join(PATH_DATA, 'onto', 'hpo-22-12-15-data')
PATH_EVALUATION = os.path.join(PATH_DATA, 'evaluation')

# Pubmed auth: required
with open(PATH_AUTH + '/pubmed/email.txt', 'r') as file:
    EMAIL = file.read().strip()
with open(PATH_AUTH + '/pubmed/api-key.txt', 'r') as file:
    PUBMED_API_KEY = file.read().strip()

# Initial BERT Models
BERTBASE =  'sentence-transformers/stsb-bert-base'
PRITAMDEKAMODEL = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'

# Misc
SEED = 42
MARGIN = 0.3743
BATCH_SIZE = 16
SAVE_BEST = False