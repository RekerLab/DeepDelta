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
import pickle



########################
### Cross-Validation ###
########################

### Random Forest ###

# Dataset loading
datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

for dataset in datasets:
  name =  dataset
  dataframe = pd.read_csv("{}.csv".format(name))
  model = RF()
  model_name = 'RF'
  
  # Process data
  mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
  fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
  dataframe["Molecule"] = mols
  dataframe["Fingerprint"] = fps
  data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})

  trad = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])

  for i in range(5): # five random states for cross-validation

    # Cross validation training of the model
    cv = KFold(n_splits=10, random_state=i, shuffle=True)
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

    # Run Stats
    pearson = stats.pearsonr(np.concatenate(trues), np.concatenate(preds))
    MAE = metrics.mean_absolute_error(np.concatenate(trues), np.concatenate(preds))
    RMSE = math.sqrt(metrics.mean_squared_error(np.concatenate(trues), np.concatenate(preds)))
    
    scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    trad = trad.append(scoring)

    # Save the actual data for use later
    datapoints = pd.DataFrame({'Y': np.concatenate(trues), 'Y_Pred': np.concatenate(preds)})
    datapoints.to_csv("{}_CV_{}_trad_{}.csv".format(name, model_name, i))

  # Save the model to be referenced later
  model.fit(np.vstack(data.Fingerprint.to_numpy()),data.Y) # First train model on the entire dataset
  filename = 'RF_control_{}.sav'.format(name)
  pickle.dump(model, open(filename, 'wb')) 
  
  # Overall Statistics
  average = pd.DataFrame({'Pearson\'s r': [round(np.mean(trad['Pearson\'s r']), 3)], 'MAE': [round(np.mean(trad['MAE']), 3)], 
                          'RMSE': [round(np.mean(trad['RMSE']), 3)]})
  std = pd.DataFrame({'Pearson\'s r': [round(np.std(trad['Pearson\'s r']), 3)], 'MAE': [round(np.std(trad['MAE']), 3)], 
                      'RMSE': [round(np.std(trad['RMSE']), 3)]})
  trad = trad.append(average)
  trad = trad.append(std)
  trad = trad.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
  trad.to_csv("{}_CV_{}_trad_Scoring.csv".format(name, model_name))
  
  
### LightGBM ###

for dataset in datasets:
  name =  dataset
  dataframe = pd.read_csv("{}.csv".format(name))
  if dataset == FUBrain:
    model = lgb(subsample=0.1, subsample_freq = 1, min_child_samples = 5) # Only for FUBrain dataset due to small size
  else:
    model = lgb(subsample=0.1, subsample_freq = 1)
  model_name = 'LGBM'
  
  # Process data
  mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
  fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
  dataframe["Molecule"] = mols
  dataframe["Fingerprint"] = fps
  data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})
  
  trad = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])

  for i in range(5): # five random states for cross-validation

    # Cross validation training of the model
    cv = KFold(n_splits=10, random_state=i, shuffle=True)
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

    # Run stats
    pearson = stats.pearsonr(np.concatenate(trues), np.concatenate(preds))
    MAE = metrics.mean_absolute_error(np.concatenate(trues), np.concatenate(preds))
    RMSE = math.sqrt(metrics.mean_squared_error(np.concatenate(trues), np.concatenate(preds)))
    
    scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    trad = trad.append(scoring)

    # Save the actual data for use later
    datapoints = pd.DataFrame({'Y': np.concatenate(trues), 'Y_Pred': np.concatenate(preds)})
    datapoints.to_csv("{}_CV_{}_trad_{}.csv".format(name, model_name, i))

  # Save the model to be referenced later
  model.fit(np.vstack(data.Fingerprint.to_numpy()),data.Y) # Train model on the entire dataset
  filename = 'LGBM_control_subsample_{}.sav'.format(name)
  pickle.dump(model, open(filename, 'wb')) # Save the model to be referenced later
  
  # Overall Statistics
  average = pd.DataFrame({'Pearson\'s r': [round(np.mean(trad['Pearson\'s r']), 3)], 'MAE': [round(np.mean(trad['MAE']), 3)], 
                          'RMSE': [round(np.mean(trad['RMSE']), 3)]})
  std = pd.DataFrame({'Pearson\'s r': [round(np.std(trad['Pearson\'s r']), 3)], 'MAE': [round(np.std(trad['MAE']), 3)], 
                      'RMSE': [round(np.std(trad['RMSE']), 3)]})
  trad = trad.append(average)
  trad = trad.append(std)
  trad = trad.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
  trad.to_csv("{}_CV_{}_trad_Scoring.csv".format(name, model_name))

########################


##########################
### External Test Sets ###
##########################

# Load datasets and models
properties = ['Sol', 'VDss']
loaded_RF_model = pickle.load(open('RF_control_{}.sav'.format(name), 'rb'))
loaded_LGBM_model = pickle.load(open('LGBM_control_subsample_{}.sav'.format(name), 'rb'))

for property in properties:
    dataset = '{}.csv'.format(property)
    pred_dataset = '{}ExternalTestSetFiltered.csv'.format(property)

    # Prepare data
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Molecule"] = mols
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})

    # Predictions - Random Forest
    data["Y_pred"] = loaded_RF_model.predict(np.vstack(data.Fingerprint.to_numpy()))  # make predictions
    pair = pd.merge(data, data, how='cross') # Cross merge the data together
    pair["Fingerprint"] =  pair.Fingerprint_x.combine(pair.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
    pair["Delta"] = pair.Y_x - pair.Y_y # calculate Delta values
    pair["Delta_pred"] = pair.Y_pred_x - pair.Y_pred_y # calculate predicted delta values
    pair.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y', 'Y_pred_x', 'Y_pred_y'], axis=1, inplace=True) # rid of unnecessary files
    pair.to_csv('{}_RF_Control_Preds_Ext.csv'.format(property), index = False) # save

    # Predictions - LGBM
    data["Y_pred"] = loaded_LGBM_model.predict(np.vstack(data.Fingerprint.to_numpy()))  # make predictions
    pair = pd.merge(data, data, how='cross') # Cross merge the data together
    pair["Fingerprint"] =  pair.Fingerprint_x.combine(pair.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
    pair["Delta"] = pair.Y_x - pair.Y_y # calculate Delta values
    pair["Delta_pred"] = pair.Y_pred_x - pair.Y_pred_y # calculate predicted delta values
    pair.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y', 'Y_pred_x', 'Y_pred_y'], axis=1, inplace=True) # rid of unnecessary files
    pair.to_csv('{}_LGBM_Control_subsample_Preds_Ext.csv'.format(property), index = False) # save
    
##########################