# Imports
import pandas as pd
import numpy as np
from sklearn import metrics

# Dataset loading:
datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']
scoring = pd.DataFrame(columns=['Dataset', 'MAE', 'RMSE'])

for dataset in datasets:
  name =  dataset
  test_set = pd.read_csv("{}.csv".format(name))
  preds = pd.read_csv("{}_DeepDelta_1.csv".format(name)).T
  preds.columns =['True', 'Delta', 'Traditional']

  # Set up for cross validation
  from sklearn.model_selection import KFold
  cv = KFold(n_splits=10, random_state=1, shuffle=True)
  datapoint_x = []
  datapoint_y = []

  # Cross validation
  for train_index, test_index in cv.split(test_set):
    train_df = test_set[test_set.index.isin(train_index)]
    test_df = test_set[test_set.index.isin(test_index)]

    pair_subset_test = pd.merge(test_df, test_df, how='cross')
    pair_subset_test["True"] = pair_subset_test.Y_x - pair_subset_test.Y_y # calculate Delta values
    pair_subset_test.drop(['Y_x','Y_y'], axis=1, inplace=True)
    datapoint_x += [pair_subset_test.SMILES_x]
    datapoint_y += [pair_subset_test.SMILES_y]
    del pair_subset_test

  paired = pd.DataFrame(data={'SMILES_x':  np.concatenate(datapoint_x), 'SMILES_y':  np.concatenate(datapoint_y)})

  trues = preds['True'].tolist()
  trues = [float(i) for i in trues]
  paired['True'] = trues

  Deltas = preds['Delta']
  Deltas = [float(i) for i in Deltas]
  paired['Delta_preds'] = Deltas
  
  # Grab the matching SMILES pairs into a separate dataframe
  final_df = pd.DataFrame(columns = ['SMILES_x', 'SMILES_y', 'True', 'Delta_preds'])
  for i in range(len(paired)):
    if paired['SMILES_x'][i] == paired['SMILES_y'][i]:
      inter_df = pd.DataFrame({'SMILES_x': [paired['SMILES_x'][i]], 'SMILES_y': [paired['SMILES_y'][i]], 'True': [paired['True'][i]], 'Delta_preds': [paired['Delta_preds'][i]]})
      final_df = pd.concat([final_df, inter_df])

  # Score matching SMILES pairs
  MAE = metrics.mean_absolute_error(final_df['True'],final_df['Delta_preds'])
  RMSE = math.sqrt(metrics.mean_squared_error(final_df['True'], final_df['Delta_preds']))
  scoring_inter = pd.DataFrame({'Dataset': [name], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})

  scoring = scoring.append(scoring_inter)

scoring.to_csv('EquationOneResults.csv', index = False)