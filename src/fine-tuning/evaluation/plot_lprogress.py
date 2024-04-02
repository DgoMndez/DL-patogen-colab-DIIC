# %%
import argparse
import pandas as pd
from matplotlib import pyplot as plt

# 1. Se recibe el path del directorio de evaluación como argumento

def move_last_row_to_first(df):
    # Obtén los índices actuales del DataFrame
    indices = df.index.tolist()

    # Mueve el último índice al principio de la lista
    indices = [indices[-1]] + indices[:-1]

    # Reindexa el DataFrame
    df = df.reindex(indices)

    # Resetea los índices del DataFrame
    df = df.reset_index(drop=True)

    return df

def preprocess_original_eval(ev_path):
    # Almacena los nombres de los archivos en una lista
    file_names = ['similarity_evaluation_train_results.csv', 'similarity_evaluation_test_results.csv', 'MSE_similarity_evaluation_train_results.csv', 'MSE_similarity_evaluation_test_results.csv']

    # Carga los datos en DataFrames y almacena los DataFrames en una lista
    dfs = [pd.read_csv(ev_path + '/' + file_name) for file_name in file_names]

    # Mover la última fila con epoch=0 step=0 al principio
    for df, file_name in zip(dfs, file_names):
        if df['steps'].iloc[-1] == 0 and df['epoch'].iloc[-1] == 0:
            df = move_last_row_to_first(df)
            df.to_csv(ev_path + '/' + file_name, index=False)

def plot_eval(ev_path, num_batches, save=False):
    # %%
    
    # 2. Obtener los y=pearson, x=steps

    dfScoreTrain = pd.read_csv(ev_path + '/similarity_evaluation_train_results.csv')
    dfScoreTest = pd.read_csv(ev_path + '/similarity_evaluation_test_results.csv')
    dfMSETrain = pd.read_csv(ev_path + '/MSE_similarity_evaluation_train_results.csv')
    dfMSETest = pd.read_csv(ev_path + '/MSE_similarity_evaluation_test_results.csv')

    def calc_batch(row):
        if row['steps'] == -1:
            return (row['epoch']+1) * num_batches
        else:
            return row['steps'] + row['epoch'] * num_batches

    for df in [dfScoreTrain, dfScoreTest, dfMSETrain, dfMSETest]:
        df['batch'] = df.apply(lambda row: calc_batch(row), axis=1)

    def get_data(df, columns):
        k = len(columns)

        x = []
        y = [[] for i in range(k)]

        for row in df.iterrows():
            if row[1]['steps'] == -1:
                x_val = (row[1]['epoch']+1) * num_batches
            else :
                x_val = row[1]['steps'] + row[1]['epoch'] * num_batches
            row[1]['batch'] = x_val
            for i in range(k):
                z = row[1][columns[i]]
                y[i].append(z)
            x.append(x_val)
        
        return x, y


    def plot_data(xtrain, ytrain, xtest, ytest, ylabel,
                  best_xtrain=None, best_ytrain=None,
                  best_xtest=None, best_ytest=None):
        plt.plot(xtrain, ytrain, label='Train')
        plt.plot(xtest, ytest, label='Test')
        xlabel = 'Batches'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ylabel + ' vs ' + xlabel)
        titleString = ylabel + ' vs ' + xlabel
        titleString.replace(' ', '_')
        # Mark best point
        if best_xtrain is not None and best_ytrain is not None:
            plt.scatter(best_xtrain, best_ytrain, color='blue')
        if best_xtest is not None and best_ytest is not None:
            plt.scatter(best_xtest, best_ytest, color='orange')
        plt.legend()
        # Save the plot to a file
        if save:
            plt.savefig(ev_path + '/' + titleString + '.png')
        else:
            plt.show()
        plt.close()

    # 2. Obtener los datos
    xtrain, ytrain = get_data(dfScoreTrain, ['cosine_pearson', 'cosine_spearman'])
    xtest, ytest = get_data(dfScoreTest, ['cosine_pearson', 'cosine_spearman'])

    # 4. Best scores
    dfBestsTrain = pd.DataFrame(columns=['metric', 'value', 'step'])
    dfBestsTest = pd.DataFrame(columns=['metric', 'value', 'step'])

    dfBests = [dfBestsTrain, dfBestsTest]
    dfScores = [dfScoreTrain, dfScoreTest]
    dfMSEs = [dfMSETrain, dfMSETest]

    for i in range(2):
        j = 0
        df = dfScores[i]
        for colname in ['cosine_pearson', 'cosine_spearman']:
            argmax = df[colname].idxmax()
            max_val = df[colname].iloc[argmax]
            stepmax = df['batch'].iloc[argmax]
            dfBests[i].loc[j] = {'metric': colname, 'value': max_val, 'step': stepmax}
            j += 1 
        df = dfMSEs[i]
        for colname in ['MSE_cosine']:
            argmin = df[colname].idxmin()
            min_val = df[colname].iloc[argmin]
            stepmin = df['batch'].iloc[argmin]
            dfBests[i].loc[j] = {'metric': colname, 'value': min_val, 'step': stepmin}
            j += 1
    
    # 3. Graficar Correlation
    # Marcando best scores
    plot_data(xtrain, ytrain[0], xtest, ytest[0], 'Pearson Correlation',
              dfBestsTrain.loc[0]['step'], dfBestsTrain.loc[0]['value'],
              dfBestsTest.loc[0]['step'], dfBestsTest.loc[0]['value'])
    plot_data(xtrain, ytrain[1], xtest, ytest[1], 'Spearman Correlation',
              dfBestsTrain.loc[1]['step'], dfBestsTrain.loc[1]['value'],
              dfBestsTest.loc[1]['step'], dfBestsTest.loc[1]['value'])

    xtrain, ytrain = get_data(dfMSETrain, ['MSE_cosine'])
    xtest, ytest = get_data(dfMSETest, ['MSE_cosine'])

    plot_data(xtrain, ytrain[0], xtest, ytest[0], 'MSE',
              dfBestsTrain.loc[2]['step'], dfBestsTrain.loc[2]['value'],
              dfBestsTest.loc[2]['step'], dfBestsTest.loc[2]['value'])

    # Devolver max scores
    return(dfBestsTrain, dfBestsTest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation plots for the evaluation of the fine-tuning process')
    parser.add_argument('--path', type=str, required=True, help='eval path in the model directory')
    parser.add_argument('--num_batches', type=int, required=True, help='number of batches per epoch (can be seen in README.md)')
    args = parser.parse_args()

    ev_path = args.path
    num_batches = args.num_batches

    preprocess_original_eval(ev_path)
    plot_eval(ev_path, num_batches, save=True)