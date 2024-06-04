# %%
# Import global variables from project_config.py

import itertools
import os
import sys
import argparse

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(src_path)

from project_config.project_config import *

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
from plot_lprogress import plot_eval
import argparse

N=1255
save=True
dir = PATH_OUTPUT + '/grid'

argparser = argparse.ArgumentParser('Plot evaluation results from grid search')
argparser.add_argument('-n', type=int, default=1255)
argparser.add_argument('-d', '--dir', type=str, default=dir)

# para cada subdirectorio en el directorio de evaluación
# llamar a plot_lprogress

# for every subdirectory in the grid directory
for modeldir in os.listdir(dir):
    # construct the evaluation path
    ev_path = os.path.join(dir, modeldir, 'eval')
    # call the plot_eval function
    plot_eval(ev_path, N, save)