import chemprop

import os

import pandas as pd
import numpy as np

from sklearn import metrics
from scipy import stats as stats
import math

from sklearn.model_selection import KFold

import tempfile
import shutil

import abc


class abstractDeltaModel(metaclass=abc.ABCMeta):
    model = None

    @abc.abstractmethod
    def fit_delta(self, x, y):
        pass

    @abc.abstractmethod
    def predict_delta(self, x):
        pass    
    
    @abc.abstractmethod
    def fit_single(self, x, y):
        pass

    @abc.abstractmethod
    def predict_single(self, x):
        pass



class Delta_ChemProp(abstractDeltaModel):
    epochs = None
    dirpath = None  # used the model paths as variables of the objects so we can always access them after fitting instead of passing back and forth through the methods
    dirpath_single = None

    def __init__(self, epochs=100, dirpath = None, dirpath_single = None):
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit_delta(self, x, y, metric='r2'):
        self.dirpath = tempfile.NamedTemporaryFile().name
        
        train = pd.merge(x, x, how='cross')
        y_values = pd.merge(y, y, how='cross')
        train["Y"] = y_values.Y_y - y_values.Y_x # calculate Delta values
        del y_values

        temp_datafile = tempfile.NamedTemporaryFile() # use temporary files for data storage
        train.to_csv(temp_datafile.name, index=False)

        arguments = [
            '--data_path', temp_datafile.name,
            '--dataset_type', 'regression', # since delta model is always regression we can keep this variable fixed
            '--save_dir', self.dirpath,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1', # fixed ensemble to 1 as well since we do not use that parameter at the moment
            '--epochs', str(self.epochs),
            '--metric', metric,
            '--number_of_molecules', '2',
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

        temp_datafile.close()

    def fit_single(self, x, y, metric='r2'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name
        
        train = pd.DataFrame(np.transpose(np.vstack([x,y])),columns=["X","Y"])

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        arguments = [
            '--data_path', temp_datafile.name,
            '--dataset_type', 'regression',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '1',
            '--metric', metric, #included metric since we have it for the delta model as well
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

        temp_datafile.close()

    def predict_delta(self, x): #modified to reverse order

        dataset = pd.merge(x, x, how='cross')

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, # TODO could possible be replaced by '/dev/null'
            '--checkpoint_dir', self.dirpath,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args)

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions

    def predict_single(self, x): 

        dataset = pd.DataFrame(x)
        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, # TODO could possible be replaced by '/dev/null'
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args)

        predictions = pd.read_csv(temp_predfile.name)['Y'] 

        preds = pd.merge(predictions,predictions,how='cross')

        temp_datafile.close()
        temp_predfile.close()

        return preds.Y_y - preds.Y_x #changed the output of predict single to also create delta values from single predictions

def cross_validation(x, y, model, k=10, random_state=1): # provide option to cross validate with x and y instead of file
  kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

  preds_single = []
  vals  = []

# ONLY DOES TRADITIONAL AND NOT DELTA

  for train, test in kf.split(x):
        model.fit_single(x[train],y[train])
        preds_single = np.append(preds_single, model.predict_single(x[test]))
        y_pairs = pd.merge(y[test],y[test],how='cross')
        vals  = np.append(vals, y_pairs.Y_y - y_pairs.Y_x)

  return [vals, preds_single]


def cross_validation_file(data_path, model, k=10, random_state=1):
  df = pd.read_csv(data_path)
  x = df[df.columns[0]]
  y = df[df.columns[1]]
  return cross_validation(x,y,model,k,random_state) # new cross validation function simplifies this one

def pred_ext(training_data_path, pred_data_path, model): # provide option to cross validate with x and y instead of file ONLY TRADITIONAL

    df = pd.read_csv(training_data_path)
    train_x = df[df.columns[0]]
    train_y = df[df.columns[1]]

    pred_df = pd.read_csv(pred_data_path)
    pred_x = pred_df[pred_df.columns[0]]
    pred_y = pred_df[pred_df.columns[1]]

    preds_single = []
    vals  = []


    model.fit_single(train_x,train_y)
    preds_single = np.append(preds_single, model.predict_single(pred_x))
    y_pairs = pd.merge(pred_y,pred_y,how='cross')
    vals  = np.append(vals, y_pairs.Y_y - y_pairs.Y_x)

    return [vals,preds_single]

###################
####  5x10 CV  ####
###################

properties = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

model = Delta_ChemProp(epochs=50)
delta = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])
trad = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])

for property in properties:
    for i in range(5):
        dataset = '{}.csv'.format(property)
        results = cross_validation_file(data_path=dataset, model=model, k=10, iter = i)

        pd.DataFrame(results).to_csv("{}_DeepDelta_50_Epoch_Trad_{}.csv".format(property, i), index=False)
        #If you .T the dataframe, then the first column is ground truth, the second is delta model, and the third is single model

        df = pd.read_csv("{}_DeepDelta_50_Epoch_Trad_{}.csv".format(property, i)).T
        df.columns =['True', 'Delta']
        preds = df['Delta'].tolist()
        trues = df['True'].tolist()

        pearson = stats.pearsonr(trues, preds)
        MAE = metrics.mean_absolute_error(trues, preds)
        RMSE = math.sqrt(metrics.mean_squared_error(trues, preds))
        scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
        delta = delta.append(scoring)

    average = pd.DataFrame({'Pearson\'s r': [round(np.mean(delta['Pearson\'s r']), 3)], 'MAE': [round(np.mean(delta['MAE']), 3)], 'RMSE': [round(np.mean(delta['RMSE']), 3)]})
    std = pd.DataFrame({'Pearson\'s r': [round(np.std(delta['Pearson\'s r']), 3)], 'MAE': [round(np.std(delta['MAE']), 3)], 'RMSE': [round(np.std(delta['RMSE']), 3)]})
    delta = delta.append(average)
    delta = delta.append(std)
    delta = delta.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
    delta.to_csv("{}_DeepDelta_50_Epoch_Trad_scoring.csv".format(property))
    
###################


#############################
####  External Test Set  ####
#############################

properties = ['Sol', 'VDss']
model = Delta_ChemProp(epochs=50)

for property in properties:
    dataset = '{}.csv'.format(property)
    pred_dataset = '{}ExternalTestSetFiltered.csv'.format(property)
    pred_name = '{}_Test'.format(property)
    
    results = pred_ext(training_data_path=dataset, pred_data_path = pred_dataset, model=model)

    pd.DataFrame(results).to_csv("{}_DeepDelta_50_Epoch_Trad_{}.csv".format(property, pred_name), index=False)
    #If you .T the dataframe, then the first column is ground truth, the second is delta model, and the third is single model

#############################