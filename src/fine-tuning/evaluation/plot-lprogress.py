# %%
import argparse
import pandas as pd
from matplotlib import pyplot as plt

# 1. Se recibe el path del directorio de evaluación como argumento

parser = argparse.ArgumentParser(description='Correlation plots for the evaluation of the fine-tuning process')
parser.add_argument('--path', type=str, required=True, help='eval path in the model directory')
parser.add_argument('--num_batches', type=int, required=True, help='number of batches per epoch (can be seen in README.md)')
args = parser.parse_args()

ev_path = args.path
num_batches = args.num_batches

# 2. Obtener los y=pearson, x=steps

dfScoreTrain = pd.read_csv(ev_path + '/similarity_evaluation_train_results.csv')
dfScoreTest = pd.read_csv(ev_path + '/similarity_evaluation_test_results.csv')


def get_data(df):
    x = []
    y = []
    z = []

    for row in df.iterrows():
        if row[1]['steps'] == -1:
            x_val = (row[1]['epoch']+1) * num_batches
        else :
            x_val = row[1]['steps'] + row[1]['epoch'] * num_batches
        y_val = row[1]['cosine_pearson']
        z_val = row[1]['cosine_spearman']
        x.append(x_val)
        y.append(y_val)
        z.append(z_val)
    
    return x, y, z

def plot_data(xtrain, ytrain, xtest, ytest, ylabel):
    plt.plot(xtrain, ytrain, label='Train')
    plt.plot(xtest, ytest, label='Test')
    xlabel = 'Batches'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel + ' vs ' + xlabel)
    titleString = ylabel + ' vs ' + xlabel
    titleString.replace(' ', '_')
    plt.legend()
    # Save the plot to a file
    plt.savefig(ev_path + '/' + titleString + '.png')
    plt.close()

# 2. Obtener los datos
xtrain, ytrain, ztrain = get_data(dfScoreTrain)
xtest, ytest, ztest = get_data(dfScoreTest)

# 3. Graficar
plot_data(xtrain, ytrain, xtest, ytest, 'Pearson Correlation')
plot_data(xtrain, ztrain, xtest, ztest, 'Spearman Correlation')