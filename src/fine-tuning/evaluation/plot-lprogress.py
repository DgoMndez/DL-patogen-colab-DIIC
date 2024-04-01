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

# %%
# 2. Obtener los y=pearson, x=steps

dfScoreTrain = pd.read_csv(ev_path + '/similarity_evaluation_train_results.csv')
dfScoreTest = pd.read_csv(ev_path + '/similarity_evaluation_test_results.csv')
dfMSETrain = pd.read_csv(ev_path + '/MSE_similarity_evaluation_train_results.csv')
dfMSETest = pd.read_csv(ev_path + '/MSE_similarity_evaluation_test_results.csv')

def get_data(df, columns):
    k = len(columns)

    x = []
    y = [[] for i in range(k)]

    for row in df.iterrows():
        if row[1]['steps'] == -1:
            x_val = (row[1]['epoch']+1) * num_batches
        else :
            x_val = row[1]['steps'] + row[1]['epoch'] * num_batches
        for i in range(k):
            z = row[1][columns[i]]
            y[i].append(z)
        x.append(x_val)
    
    return x, y


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
xtrain, ytrain = get_data(dfScoreTrain, ['cosine_pearson', 'cosine_spearman'])
xtest, ytest = get_data(dfScoreTest, ['cosine_pearson', 'cosine_spearman'])

# 3. Graficar Correlation
plot_data(xtrain, ytrain[0], xtest, ytest[0], 'Pearson Correlation')
plot_data(xtrain, ytrain[1], xtest, ytest[1], 'Spearman Correlation')

xtrain, ytrain = get_data(dfMSETrain, ['MSE_cosine'])
xtest, ytest = get_data(dfMSETest, ['MSE_cosine'])

plot_data(xtrain, ytrain[0], xtest, ytest[0], 'MSE')