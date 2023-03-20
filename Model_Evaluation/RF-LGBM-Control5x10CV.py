# Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RF
from lightgbm import LGBMRegressor as lgb


#####################
### Random Forest ###
#####################

# Dataset loading:
datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

all_scores = pd.DataFrame(columns=['Dataset', 'Pearson\'s r', 'MAE', 'RMSE'])
prism_data = pd.DataFrame(columns=['Dataset', 'Pearson\'s r Mean', 'Pearson\'s r STD', 'r N', 'MAE Mean', 'MAE STD', 'MAE N', 'RMSE Mean', 'RMSE STD', 'RMSE N'])

for dataset in datasets:
  name =  dataset
  dataframe = pd.read_csv("{}.csv".format(name))

  # Process data
  mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
  fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
  dataframe["Molecule"] = mols
  dataframe["Fingerprint"] = fps
  data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})

  # Model 1 Random Forest
  model = RF()
  model_name = 'RF'
  
  trad = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])

  for i in range(5):

    # Set up for cross validation
    cv = KFold(n_splits=10, random_state=i, shuffle=True)

    # Cross validation training of the model
    preds = []
    trues = []

    for train_index, test_index in cv.split(data):
        train_df = data[data.index.isin(train_index)]
        test_df = data[data.index.isin(test_index)]
        
        model.fit(np.vstack(train_df.Fingerprint.to_numpy()),train_df.Y) # fit model on pair training data
        test_df["Y_pred"] = model.predict(np.vstack(test_df.Fingerprint.to_numpy()))  # make predictions

        pair_subset_test = pd.merge(test_df, test_df, how='cross') # Cross merge the data together
        pair_subset_test["Fingerprint"] =  pair_subset_test.Fingerprint_x.combine(pair_subset_test.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
        pair_subset_test["Delta"] = pair_subset_test.Y_x - pair_subset_test.Y_y # calculate Delta values
        pair_subset_test["Delta_pred"] = pair_subset_test.Y_pred_x - pair_subset_test.Y_pred_y # calculate predicted delta values
        pair_subset_test.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True) # rid of unnecessary files

        trues += [pair_subset_test.Delta]
        preds += [pair_subset_test.Delta_pred]
        del pair_subset_test # rid of unnecessary files

    pearson = stats.pearsonr(np.concatenate(trues), np.concatenate(preds))
    MAE = metrics.mean_absolute_error(np.concatenate(trues), np.concatenate(preds))
    RMSE = math.sqrt(metrics.mean_squared_error(np.concatenate(trues), np.concatenate(preds)))
    
    scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    trad = trad.append(scoring)

    # Save the actual data for use later
    datapoints = pd.DataFrame({'Y': np.concatenate(trues), 'Y_Pred': np.concatenate(preds)})
    datapoints.to_csv("{}_CV_{}_trad_{}.csv".format(name, model_name, i))
    files.download("{}_CV_{}_trad_{}.csv".format(name, model_name, i))


  # Statistics for each round
  average = pd.DataFrame({'Pearson\'s r': [round(np.mean(trad['Pearson\'s r']), 3)], 'MAE': [round(np.mean(trad['MAE']), 3)], 
                          'RMSE': [round(np.mean(trad['RMSE']), 3)]})
  std = pd.DataFrame({'Pearson\'s r': [round(np.std(trad['Pearson\'s r']), 3)], 'MAE': [round(np.std(trad['MAE']), 3)], 
                      'RMSE': [round(np.std(trad['RMSE']), 3)]})
  trad = trad.append(average)
  trad = trad.append(std)
  trad = trad.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
  trad.to_csv("{}_CV_{}_trad_Scoring.csv".format(name, model_name))
  files.download("{}_CV_{}_trad_Scoring.csv".format(name, model_name))

  # Make Summary Statistics Easy to put into a table 
  # However, there is a random odd character that shows up before the ±.
  # Use CTRL+H to open Find and Replace and then search for that character and use Ctrl + J to enter a line break so the table is formatted nicely
  both = pd.DataFrame({'Dataset': name, 
                    'Pearson\'s r': average["Pearson\'s r"].astype(str) + "±" + std["Pearson\'s r"].astype(str), 
                    'MAE': average["MAE"].astype(str) + "±" + std["MAE"].astype(str), 
                    'RMSE': average["RMSE"].astype(str) + "±" + std["RMSE"].astype(str)})
  all_scores = all_scores.append(both)

  # Make summary statistics easy to implemented into Prism
  prism = pd.DataFrame({'Dataset': name, 
                    'Pearson\'s r Mean': average["Pearson\'s r"], 
                    'Pearson\'s r STD': std["Pearson\'s r"],
                    'r N': '5',
                    'MAE Mean': average["MAE"], 
                    'MAE STD': std["MAE"],
                    'MAE N': '5', 
                    'RMSE Mean': average["RMSE"], 
                    'RMSE STD': std["RMSE"],
                    'RMSE N': '5'})
  prism_data = prism_data.append(prism)





################
### LightGBM ###
################

# Dataset loading:
datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

all_scores = pd.DataFrame(columns=['Dataset', 'Pearson\'s r', 'MAE', 'RMSE'])
prism_data = pd.DataFrame(columns=['Dataset', 'Pearson\'s r Mean', 'Pearson\'s r STD', 'r N', 'MAE Mean', 'MAE STD', 'MAE N', 'RMSE Mean', 'RMSE STD', 'RMSE N'])


for dataset in datasets:
  name =  dataset
  dataframe = pd.read_csv("{}.csv".format(name))

  # Process data
  mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
  fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
  dataframe["Molecule"] = mols
  dataframe["Fingerprint"] = fps
  data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})
  
  # Option 2 Microscoft LightGBM
  if dataset == FUBrain:
    model = lgb(subsample=0.1, subsample_freq = 1, min_child_samples = 5) # Only for FUBrain dataset due to small size
  else:
    model = lgb(subsample=0.1, subsample_freq = 1)
  model_name = 'LGBM'

  trad = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])

  for i in range(5):

    # Set up for cross validation
    cv = KFold(n_splits=10, random_state=i, shuffle=True)

    # Cross validation training of the model
    preds = []
    trues = []

    for train_index, test_index in cv.split(data):
        train_df = data[data.index.isin(train_index)]
        test_df = data[data.index.isin(test_index)]
        
        model.fit(np.vstack(train_df.Fingerprint.to_numpy()),train_df.Y) # fit model on pair training data
        test_df["Y_pred"] = model.predict(np.vstack(test_df.Fingerprint.to_numpy()))  # make predictions

        pair_subset_test = pd.merge(test_df, test_df, how='cross') # Cross merge the data together
        pair_subset_test["Fingerprint"] =  pair_subset_test.Fingerprint_x.combine(pair_subset_test.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
        pair_subset_test["Delta"] = pair_subset_test.Y_x - pair_subset_test.Y_y # calculate Delta values
        pair_subset_test["Delta_pred"] = pair_subset_test.Y_pred_x - pair_subset_test.Y_pred_y # calculate predicted delta values
        pair_subset_test.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True) # rid of unnecessary files

        trues += [pair_subset_test.Delta]
        preds += [pair_subset_test.Delta_pred]
        del pair_subset_test # rid of unnecessary files

    pearson = stats.pearsonr(np.concatenate(trues), np.concatenate(preds))
    MAE = metrics.mean_absolute_error(np.concatenate(trues), np.concatenate(preds))
    RMSE = math.sqrt(metrics.mean_squared_error(np.concatenate(trues), np.concatenate(preds)))
    
    scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    trad = trad.append(scoring)

    # Save the actual data for use later
    datapoints = pd.DataFrame({'Y': np.concatenate(trues), 'Y_Pred': np.concatenate(preds)})
    datapoints.to_csv("{}_CV_{}_trad_{}.csv".format(name, model_name, i))
    files.download("{}_CV_{}_trad_{}.csv".format(name, model_name, i))


  # Statistics for each round
  average = pd.DataFrame({'Pearson\'s r': [round(np.mean(trad['Pearson\'s r']), 3)], 'MAE': [round(np.mean(trad['MAE']), 3)], 
                          'RMSE': [round(np.mean(trad['RMSE']), 3)]})
  std = pd.DataFrame({'Pearson\'s r': [round(np.std(trad['Pearson\'s r']), 3)], 'MAE': [round(np.std(trad['MAE']), 3)], 
                      'RMSE': [round(np.std(trad['RMSE']), 3)]})
  trad = trad.append(average)
  trad = trad.append(std)
  trad = trad.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
  trad.to_csv("{}_CV_{}_trad_Scoring.csv".format(name, model_name))
  files.download("{}_CV_{}_trad_Scoring.csv".format(name, model_name))

  # Make Summary Statistics Easy to put into a table 
  # However, there is a random odd character that shows up before the ±.
  # Use CTRL+H to open Find and Replace and then search for that character and use Ctrl + J to enter a line break so the table is formatted nicely
  both = pd.DataFrame({'Dataset': name, 
                    'Pearson\'s r': average["Pearson\'s r"].astype(str) + "±" + std["Pearson\'s r"].astype(str), 
                    'MAE': average["MAE"].astype(str) + "±" + std["MAE"].astype(str), 
                    'RMSE': average["RMSE"].astype(str) + "±" + std["RMSE"].astype(str)})
  all_scores = all_scores.append(both)

  # Make summary statistics easy to implemented into Prism
  prism = pd.DataFrame({'Dataset': name, 
                    'Pearson\'s r Mean': average["Pearson\'s r"], 
                    'Pearson\'s r STD': std["Pearson\'s r"],
                    'r N': '5',
                    'MAE Mean': average["MAE"], 
                    'MAE STD': std["MAE"],
                    'MAE N': '5', 
                    'RMSE Mean': average["RMSE"], 
                    'RMSE STD': std["RMSE"],
                    'RMSE N': '5'})
  prism_data = prism_data.append(prism)