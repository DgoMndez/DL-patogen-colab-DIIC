# %% [markdown]

# # Learning progress: Resultados de Evaluación del BERT a lo largo del finetuning
# # Grid de hiperparámetros para 2ª iteración 

# ## 1. Carga de datos

# %%
# Import global variables from project_config.py

import itertools
import os
import sys
import argparse

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_path)

from project_config.project_config import *
print(f"SEED={SEED}")

# %%
# IMPORTS

from cmath import nan
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

from pyhpo import Ontology

from sentence_transformers.evaluation import SimilarityFunction
import MSESimilarityEvaluator as MSESim
from MSESimilarityEvaluator import MSESimilarityEvaluator
import time

from matplotlib import pyplot as plt

torch.manual_seed(SEED)
print(torch.cuda.is_available())

# ARGS
# MARGIN, lr, weight_decay, warmup_steps_frac, num_epochs, save_best, output
parser = argparse.ArgumentParser(description='Grid search for BERT finetuning')
parser.add_argument('-M', '--margin', type=float, nargs='+', default=[0.3743, 0.6829, 0.8974], help='List of margins for triplet loss')
parser.add_argument('--lr', type=float, nargs='+', default=[1e-07, 5e-06], help='List of learning rates')
parser.add_argument('--wd', type=float, nargs='+', default=[0, 0.005], help='List of weight decays')
parser.add_argument('--wsf', type=float, nargs='+', default=[1, 5], help='List of warmup steps fractions to try')
parser.add_argument('-e', '--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
parser.add_argument('-p', '--percent', type=float, default=0.1, help='Percent of training data to use for profiling')
parser.add_argument('-s', '--steps', type=float, default=5, help='Number of evaluation steps per epoch')
parser.add_argument('--eval_percent', type=float, default = 20, help='Percent of evaluation pairs used')
parser.add_argument('--save_best', type=bool, default=True, help='Save best model (0/1)')
parser.add_argument('-o', '--output', type=str, default='grid/fine-tuned-bio-bert', help='Output name')
parser.add_argument('--download', action='store_true', help='Download BERT model')

margins = parser.parse_args().margin
lrs = parser.parse_args().lr
wds = parser.parse_args().wd
wsfs = parser.parse_args().wsf
num_epochs = parser.parse_args().epochs
f_samp = parser.parse_args().percent / 100
ev_samp = parser.parse_args().eval_percent / 100 
output_name = parser.parse_args().output
SAVE_BEST = parser.parse_args().save_best
steps_epoch = parser.parse_args().steps
download = parser.parse_args().download

if f_samp > 0:
    PROFILING = True

if ev_samp > 0:
    EV_PROFILING = True

param_grid = {
    'margin': margins,
    'lr': lrs,
    'wd': wds,
    'wsf': wsfs
}

# %%
# 0. GPU
import torch

init_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.cuda.empty_cache()

# %%
# 1. Cargar todos los datos (crudos)

# 1.1 BERT de partida

MAX_SEQ_LENGTH = 256

PATH_BASE = os.path.join(PATH_OUTPUT, 'base')

if download:
    bertmodel = SentenceTransformer(PRITAMDEKAMODEL, device=device) # Original
    bertmodel.save(PATH_BASE)
else:
    bertmodel = SentenceTransformer(PATH_BASE)

model = bertmodel # Para finetunear

# 1.2 Ontología

onto = Ontology(PATH_ONTO)

# %%
# 1.3 Datos de entrenamiento y evaluación (csv)

# abstracts (train)
path_abstracts_train = os.path.join(PATH_ABSTRACTS, 'abstracts-31-05-train.csv')
dTrain = pd.read_csv(path_abstracts_train, sep='\t', low_memory=False, na_values=['', nan])

if PROFILING:
    dTrain = dTrain.sample(frac=f_samp, random_state=SEED, ignore_index=True)

print(f"N={dTrain.shape[0]} abstracts")

# %%
# fenotipos hoja
path_phenotypes = os.path.join(PATH_PHENOTYPES, 'phenotypes_nz_05-05_10.csv')
dfPhenotypes = pd.read_csv(path_phenotypes, sep='\t', low_memory=False, na_values=['', nan])


# fenotipos tags
path_index = os.path.join(PATH_PHENOTYPES, 'index-31-05-train.csv')
dfIndex = pd.read_csv(path_index, sep='\t', low_memory=False, na_values=['', nan])
print(f"m={dfIndex.shape[0]} tags")

# %%
# pares fenotipos train
path_pairs_train = os.path.join(PATH_EVALUATION, 'train_pairs-31-05.csv')
dfVal = pd.read_csv(path_pairs_train, sep='\t', low_memory=False, na_values=['', nan])

# test
path_pairs_test = os.path.join(PATH_EVALUATION, 'test_pairs-31-05.csv')
dfTest = pd.read_csv(path_pairs_test, sep='\t', low_memory=False, na_values=['', nan])

if EV_PROFILING:
    dfVal = dfVal.sample(frac=ev_samp, random_state=SEED, ignore_index=True)
    dfTest = dfTest.sample(frac=ev_samp,random_state=SEED, ignore_index=True)

print(dfVal.info())
print(dfTest.info())

# %% [markdown]
# ## 2. Preparación del entrenamiento
# Vamos a construir los dataloaders y las funciones de pérdida y evaluación.

# %% [markdown]
# ### Dataloaders

# %%
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, evaluation, InputExample

# 2.1. Ejemplos de entrenamiento
tags = dfIndex['phenotypeName']
numlabels = len(tags)
mapping = {tag: i for i, tag in enumerate(tags)}

def getLabelNumber(phenotypeName):
    return mapping[phenotypeName]

BATCH_SIZE = 16
abstractsTrain = [InputExample(texts=[x], label=mapping[y]) for x, y in zip(dTrain['clean_abstract'], dTrain['phenotypeName'])]
train_dataloader = DataLoader(abstractsTrain, shuffle=True, batch_size=BATCH_SIZE)
print("batch_size=", BATCH_SIZE)

# 2.2. Ejemplos de validación

ltrain1 = dfVal['phenotype1']
ltrain2 = dfVal['phenotype2']
goldTrain = dfVal['lin']

# 2.3. Ejemplos de test

ltest1 = dfTest['phenotype1']
ltest2 = dfTest['phenotype2']
goldTest = dfTest['lin']

# 2.5 Evaluation

evaluatorTrain1=sentence_transformers.evaluation.EmbeddingSimilarityEvaluator(ltrain1, ltrain2, goldTrain,
                                                                             main_similarity=SimilarityFunction.COSINE,
                                                                             name='train')
evaluatorTrain2=MSESimilarityEvaluator(ltrain1, ltrain2, goldTrain, main_similarity=SimilarityFunction.COSINE,name='train')
evaluatorTest1=sentence_transformers.evaluation.EmbeddingSimilarityEvaluator(ltest1, ltest2, goldTest,
                                                                            main_similarity=SimilarityFunction.COSINE,
                                                                            name='test')
evaluatorTest2=MSESimilarityEvaluator(ltest1, ltest2, goldTest, main_similarity=SimilarityFunction.COSINE,name='test')
combined_evaluator = evaluation.SequentialEvaluator([evaluatorTrain1, evaluatorTrain2, evaluatorTest1, evaluatorTest2],
                                                    main_score_function=lambda scores: scores[2])

# %%
scoreTrain = evaluatorTrain1.__call__(model=bertmodel, output_path='./results/original', epoch=0, steps=0)
scoreTest = evaluatorTest1.__call__(model=bertmodel, output_path='./results/original', epoch=0, steps=0)
print(f'Original score (spearman): {scoreTrain} (train), {scoreTest} (test)')

# CSV to save all results
path_scores_csv = PATH_OUTPUT+f'/best_scores-{pd.Timestamp("today").strftime("%d-%m-%Y")}.csv'
if not os.path.exists(path_scores_csv):
    with open(path_scores_csv, 'w') as f:
        f.write("BERTNAME,train_spearman,test_spearman, train_pearson, test_pearson, train_MSE, test_MSE, time\n")

# Grid search
best_score = -1
best_params = None
    
param_combinations = list(itertools.product(*param_grid.values()))

i = 0
j = 0
for params in param_combinations:

    params_dict = dict(zip(param_grid.keys(), params))
    print(f"Comb {i}: {params_dict}")

    # %% [markdown]
    # ### Hiperparámetros
    # [hiperparams.csv](./hiperparams.csv)

    # %%

    MARGIN = params_dict['margin']
    NUM_EPOCHS = num_epochs
    STEPS = steps_epoch
    WARMUP_STEPS_FRAC = params_dict['wsf']
    num_batches = len(train_dataloader)
    lr = params_dict['lr']
    wd = params_dict['wd']

    BERTNAME = output_name + f'-{i}-MARGIN={MARGIN}-lr={lr}-wd={wd}-wsf={WARMUP_STEPS_FRAC}'
    output_path = os.path.join(PATH_OUTPUT, BERTNAME + '-' + pd.Timestamp("today").strftime("%d-%m-%Y"))
    print(f'Output path: {output_path}')
    print(f'Hiperparams: N={num_batches}, lr={lr}, wd={wd} NUM_EPOCHS={NUM_EPOCHS}, STEPS={STEPS}, WARMUP_STEPS_FRAC={WARMUP_STEPS_FRAC}, MARGIN={MARGIN}, BERTNAME={BERTNAME}')

    ev_steps = num_batches // STEPS
    warmup_steps = num_batches // WARMUP_STEPS_FRAC

    model = SentenceTransformer(PATH_BASE)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("max_seq_length = ", model.get_max_seq_length())

    # %% [markdown]
    # ### Funciones de pérdida y evaluación
    # El código de MSESimilarityEvaluator puede verse en [MSESimilarityEvaluator.py](./MSESimilarityEvaluator.py)

    # %%
    # 2.4 Loss

    train_loss = losses.BatchAllTripletLoss(model=model, distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance, margin=MARGIN)

    # %% [markdown]
    # ## 3. Fit

    # %%
    print("Fitting...")
    import time
    FITTED = False
    if FITTED:
        DATE = "1-06-2024"
        output_path = os.path.join(PATH_OUTPUT, BERTNAME+'-'+DATE)
        model = SentenceTransformer(output_path)
        with open(os.path.join(PATH_OUTPUT, BERTNAME+'-'+DATE, 'eval', 'time.txt'), 'r') as f:
            s = f.read()
            execution_time = float(s.split()[0])
    else:
        start_time = time.time()
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=combined_evaluator,
            epochs=num_epochs,
            optimizer_params = {'lr':lr},
            weight_decay=wd,
            evaluation_steps=ev_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=SAVE_BEST,
            checkpoint_path='./checkpoint',
            checkpoint_save_steps=ev_steps,
            checkpoint_save_total_limit=num_epochs
        )

        end_time = time.time()
        execution_time = end_time - start_time

    print(f"Execution time for model.fit: {execution_time:.2f} seconds")
    if not SAVE_BEST:
        model.save(output_path)
    fmodel = model # finetuned model

    # %% [markdown]
    # ## 4. Evaluación
    # Durante el entrenamiento se van guardando los resultados de evaluación en cada evaluation step en:
    # * Correlaciones Train: [similarity_evaluation_train_results.csv](../../../output/fine-tuned-bio-bert-ev-mse-01-04-2024/eval/similarity_evaluation_train_results.csv)
    # * Correlaciones Test: [similarity_evaluation_test_results.csv](../../../output/fine-tuned-bio-bert-ev-mse-01-04-2024/eval/similarity_evaluation_test_results.csv)
    # * MSE Train: [MSE_similarity_evaluation_train_results.csv](../../../output/fine-tuned-bio-bert-ev-mse-01-04-2024/eval/MSE_similarity_evaluation_train_results.csv)
    # * MSE Test: [MSE_similarity_evaluation_test_results.csv](../../../output/fine-tuned-bio-bert-ev-mse-01-04-2024/eval/MSE_similarity_evaluation_test_results.csv)

    # %%
    from plot_lprogress import plot_eval

    path_eval = os.path.join(output_path, 'eval')

    dfScoreTrain, dfScoreTest = plot_eval(path_eval, num_batches, save=False)
    print('Train best scores:\n', dfScoreTrain)
    print('Test best scores:\n', dfScoreTest)

    bests = {'cosine_pearson': [], 'cosine_spearman': [], 'MSE_cosine' : []}

    for metric in ['cosine_pearson', 'cosine_spearman', 'MSE_cosine']:
        for df in [dfScoreTrain, dfScoreTest]:
            bests[metric].append(df.where(df['metric'] == metric)['value'].max())

    #"BERTNAME,train_spearman,test_spearman, train_pearson, test_pearson, train_MSE, test_MSE, time\n"

    # append best scores for each param combination to the csv
    with open(path_scores_csv, 'a') as f:
        rows = f"{BERTNAME},{bests['cosine_spearman'][0]},{bests['cosine_spearman'][1]},{bests['cosine_pearson'][0]},{bests['cosine_pearson'][1]},{bests['MSE_cosine'][0]},{bests['MSE_cosine'][1]},{execution_time}\n"
        f.write(rows)
        print(rows)

    if bests['cosine_spearman'][1] > best_score:
        best_score = bests['cosine_spearman'][1]
        best_params = params_dict
        j = i
    
    i += 1

total_time = time.time() - init_time
print(f"Total time: {total_time:.2f} seconds")

print(f"Best comb: {j}, score: {best_score}, params: {best_params}")