# %% [markdown]
# # Experimento finetuning 1ª iteración
# * Objetivo: Determinar si se consigue aprendizaje con el fine-tuning supervisado
# * Método: Fine-tuning del tipo (etiqueta=fenotipo, valor=abstract) con una capa softmax al final del BERT
# * Datos: abstracts.csv, index-phenotypes.csv, phenotypes-22-12-15.csv
# 
# ## 1. Cargar datos

# %%
from cmath import nan
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, evaluation, InputExample

# Cargar el BERT de partida

BERTBASE =  'sentence-transformers/stsb-bert-base'
PRITAMDEKAMODEL = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model = bertmodel = SentenceTransformer(PRITAMDEKAMODEL)
# Se puede aumentar max_seq_length? Suponemos que no

# Función clean abstract

# Download the stopwords from NLTK
nltk.download('punkt')
nltk.download('stopwords')

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

# Obtener los datos de entrenamiento

PATH_DATA = '../pubmed-queries/abstracts'
PATH_DATA_CSV = PATH_DATA + '/abstracts.csv'
PATH_DATA_FENOTIPOS = '../pubmed-queries/results/phenotypes-22-12-15.csv'
PATH_INDEX_FENOTIPOS = PATH_DATA + '/index-phenotypes.csv'
SEED = 42

dfPapers = pd.read_csv(PATH_DATA_CSV, sep='\t', low_memory=False, na_values=['', nan])
dfPhenotypes = pd.read_csv(PATH_DATA_FENOTIPOS, sep=';', low_memory=False, na_values=['', nan])
dfIndex = pd.read_csv(PATH_INDEX_FENOTIPOS, sep='\t', low_memory=False, na_values=['', nan])

# Cargar la ontología

from pyhpo import Ontology

onto = Ontology('../pubmed-queries/hpo-22-12-15-data')

# %% [markdown]
# ## 2. Obtener dataset de entrenamiento

# %%
# phenotypeId	phenotypeName	numberPapers	paperList
from itertools import combinations

# Tomar la lista de fenotipos = tags
tags = dfIndex['phenotypeName']
numlabels = len(tags)
print(numlabels, 'tags')
print(tags[:5])

# Tomar muestra aleatoria de pares de fenotipos
MARGIN = 0.3743
if MARGIN == 0: # Estimar un margin apropiado
    unique_pairs = combinations(dfIndex['phenotypeName'].drop_duplicates(), 2)
    df_pairs = pd.DataFrame(unique_pairs, columns=['phenotype1', 'phenotype2']).sample(frac=0.2, random_state=SEED)
    print('Unique pairs:', len(df_pairs))
    print('Pair 1:', df_pairs.iloc[0])
    #df_pairs['distance']=df_pairs.apply(lambda x: float(losses.SiameseDistanceMetric.COSINE_DISTANCE(torch.from_numpy(model.encode(x['phenotype1'])), torch.from_numpy(model.encode(x['phenotype2'])))), axis=1)
    df_pairs['distance'] = df_pairs.apply(lambda x: 1-util.cos_sim(model.encode(x['phenotype1']), model.encode(x['phenotype2'])), axis=1)
    margin = min(df_pairs['distance']).numpy()[0][0]
    print('Margin:', margin)

# Separar abstracts en train, validation y test

# quitar NA's en la columna abstract
print('Na\'s:', dfPapers['abstract'].isna().sum())
dfPapers = dfPapers.dropna(subset=['abstract'])

train = dfPapers.sample(frac=0.1, random_state=SEED)

dTest = dfPapers.drop(train.index).sample(frac=0.2, random_state=SEED)
dVal = train.sample(frac=0.2, random_state=SEED)
dTrain = train.drop(dVal.index)
num_examples = len(dTrain)
print(num_examples,'examples in train')

# Considerar train_test_split

# paperId	phenotypeId	phenotypeName	title	abstract
list = [dTrain, dVal, dTest]
names = ['Train', 'Validation', 'Test']
for j in range(0, 3):
    l = list[j]
    print(names[j],': ', len(l), '\n')
    for i in range(0, 2):
        print(l.iloc[i])
    print('')


# %% [markdown]
# ## 3. ¿Cómo se hace el fine-tuning?
# Para nuestro caso particular necesitamos pasarle los tags, añadir la red neuronal a la salida y la capa softmax y la forma de evaluación.
# 

# %%
torch.manual_seed(SEED)

num_epochs = 1

model = bertmodel

mapping = {tag: i for i, tag in enumerate(tags)}


def getLabelNumber(phenotypeName):
    return mapping[phenotypeName]

# %%
# TODO: Documentarse cómo se prepara el DataLoader con los pares abstract-fenotipo
# imagino que en el conjunto de train solo se usan los abstracts y en el conjunto de validación y test se usan los abstracts y los fenotipos

print("Preparing dataloaders...")

print('Cleaning abstracts...')
print('example:', clean_abstract(dTrain['abstract'].iloc[0]))

abstractsTrain = [InputExample(texts=[clean_abstract(x)], label=mapping[y]) for x, y in zip(dTrain['abstract'], dTrain['phenotypeName'])]
train_dataloader = DataLoader(abstractsTrain, shuffle=True, batch_size=16)

print('Validation')
pairsVal = [InputExample(texts=[clean_abstract(x)], label=mapping[y]) for x, y in zip(dTrain['abstract'], dVal['phenotypeName'])]
val_dataloader = DataLoader(pairsVal, shuffle=False, batch_size=16)

print('Test')
pairsTest = [InputExample(texts=[clean_abstract(x)], label=mapping[y]) for x, y in zip(dTrain['abstract'], dTest['phenotypeName'])]
test_dataloader = DataLoader(dTest, shuffle=False, batch_size=16)

# TODO: Documentarse sobre loss y evaluator

print("Preparing loss and evaluator...")
soft_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=numlabels)
# Esta no sirve porque recibe un par de sentencias y un label, no una sentencia y un label
train_loss = losses.BatchAllTripletLoss(model=model, distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance, margin=MARGIN)

evaluator = evaluation.LabelAccuracyEvaluator(val_dataloader, '', softmax_model=soft_loss, write_csv=True)


# TODO: Documentarse sobre los hiperparámetros y preparar el grid

print("Fitting...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    #evaluator=evaluator,
    epochs=num_epochs,
    #evaluation_steps=4,
    warmup_steps=int(0.25*(num_examples//16)),
    output_path='./output/fine-tuned-bio-bert',
    save_best_model=True,
    checkpoint_path='./checkpoint',
    checkpoint_save_steps=25,
    checkpoint_save_total_limit=5
)

# %%



