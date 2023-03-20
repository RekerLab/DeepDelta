# Imports
import pandas as pd
import numpy as np
from sklearn import metrics
import math
from scipy import stats as stats


####################
### Equation One ###    DeepDelta(x,x)= 0  
####################

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

scoring.to_csv('Equation_One_Results.csv', index = False)


####################
### Equation Two ###    DeepDelta(x,y)= -DeepDelta(y,x).
####################

scores = pd.DataFrame(columns=['Dataset', 'Pearson\'s r', 'MAE', 'RMSE'])

for dataset in datasets:
    name =  dataset
    df1 = pd.read_csv("{}_DeepDelta_1.csv".format(name)).T
    df2 = pd.read_csv("{}_DeepDelta_Reverse_1.csv".format(name)).T
    df1.columns =['Trues', 'Preds', 'Single']
    df2.columns =['Trues', 'Preds', 'Single']
    preds = df1['Preds'].tolist()
    preds_rev = df2['Preds'].tolist()
    
    # Statistics
    pearson = stats.pearsonr(preds, preds_rev)
    MAE = metrics.mean_absolute_error(preds,preds_rev)
    RMSE = math.sqrt(metrics.mean_squared_error(preds, preds_rev))
    scoring = pd.DataFrame({'Dataset': [name], 'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    scores = scores.append(scoring)

scores.to_csv('Equation_Two_Results.csv', index = False)



######################
### Equation Three ###     DeepDelta(x,y) + DeepDelta(y,z)= DeepDelta(x,z)
######################

for dataset in datasets:
    name =  dataset
    dataset = '{}.csv'.format(name)
    results_df = pd.DataFrame(columns=['Fold', 'Average', 'Std Dev'])
    overall_error = pd.DataFrame(columns=['Error'])

    for a in range(len(properties)):
      cnt = a+1
      full_data = pd.read_csv("{}_DeepDelta_EachCVFold_All_{}.csv".format(name, cnt)) # Requires results from each fold
      df = pd.DataFrame(index=range(len(full_data['SMILES_x'].unique())),columns=range(len(full_data['SMILES_x'].unique())))

      # Matrix
      for i in range(len(full_data['SMILES_x'].unique())):
        for j in range(len(full_data['SMILES_x'].unique())):
          position = i*len(full_data['SMILES_x'].unique()) + j
          df.iat[i, j] =  full_data['Delta_preds'][position]

      new_df = pd.DataFrame(columns = ['Error'])

      #Traverse the matrix making all the calculations (
      for i in range(len(df[0])): # Main column
        for j in range(len(df[0])): # Main row
          for k in range(len(df[0])): # Secondary column/row movement
            error = round(abs(df.iat[i,j] - (df.iat[i,k] + df.iat[k,j])), 4)
            inter_df = pd.DataFrame({'Error': [error]})
            new_df = pd.concat([new_df, inter_df])
            overall_error = pd.concat([overall_error, inter_df])

      int_df = pd.DataFrame({'Fold': [a], 'Average': [round(new_df.mean(), 4)], 'Std Dev': [round(new_df.std(), 4)]})
      results_df = results_df.append(int_df)
      print("{}/10 done.".format(cnt))

    overall_df = pd.DataFrame({'Fold': ['Overall'], 'Average': [round(overall_error.mean(), 4)], 'Std Dev': [round(overall_error.std(), 4)]})
    results_df = results_df.append(overall_df)

    pd.DataFrame(results_df).to_csv("{}_Equation_Three.csv".format(name), index=False)