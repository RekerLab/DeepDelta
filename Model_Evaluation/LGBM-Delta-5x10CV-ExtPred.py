# Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import math
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold


########################
### Cross-Validation ###
########################

# Dataset loading:
datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

for dataset in datasets:
    name =  dataset
    dataframe = pd.read_csv("{}.csv".format(name))

    # Process data
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Molecule"] = mols
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})
    del dataframe

    # Model loading
    from lightgbm import LGBMRegressor as lgb
    model = lgb(subsample=0.1, subsample_freq = 1)

    # Dataframe to store the results
    delta = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE'])

    for iter in range(5): # five random states for cross-validation

        # Cross validation training of the model
        cv = KFold(n_splits=10, random_state=iter, shuffle=True)
        preds = []
        trues = []
        cnt = 0.0
        
        for train_index, test_index in cv.split(data):
            cnt += 1
            train_df = data[data.index.isin(train_index)]
            test_df = data[data.index.isin(test_index)]

            pair_subset_train = pd.merge(train_df, train_df, how='cross')
            pair_subset_train["Fingerprint"] =  pair_subset_train.Fingerprint_x.combine(pair_subset_train.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
            pair_subset_train["Delta"] = pair_subset_train.Y_x - pair_subset_train.Y_y # calculate Delta values
            pair_subset_train.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True)

            pair_subset_test = pd.merge(test_df, test_df, how='cross')
            pair_subset_test["Fingerprint"] =  pair_subset_test.Fingerprint_x.combine(pair_subset_test.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
            pair_subset_test["Delta"] = pair_subset_test.Y_x - pair_subset_test.Y_y # calculate Delta values
            pair_subset_test.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True)

            model.fit(np.vstack(pair_subset_train.Fingerprint.to_numpy()),pair_subset_train.Delta) # fit model on pair training data
            del pair_subset_train
            preds += [model.predict(np.vstack(pair_subset_test.Fingerprint.to_numpy()))] #predict 
            trues += [pair_subset_test.Delta]
            del pair_subset_test
            print(" Finished " + str(cnt / 10 * 100) + "% of round " + str(iter) )
            

        # Export the csv from cross validation
        Trues = np.concatenate(trues)
        Preds = np.concatenate(preds)
        dict = {'trues': Trues, 'preds': Preds}     
        dataframe = pd.DataFrame(dict)
        dataframe.to_csv('{}_CV_LGBM_Delta_{}.csv'.format(name, iter), index = False)   

        # Calculate stats for each round
        pearson = stats.pearsonr(Trues, Preds)
        MAE = metrics.mean_absolute_error(Trues, Preds)
        RMSE = math.sqrt(metrics.mean_squared_error(Trues, Preds))
        scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
        delta = delta.append(scoring)

    # Calculate overall stats
    average = pd.DataFrame({'Pearson\'s r': [round(np.mean(delta['Pearson\'s r']), 3)], 'MAE': [round(np.mean(delta['MAE']), 3)], 'RMSE': [round(np.mean(delta['RMSE']), 3)]})
    std = pd.DataFrame({'Pearson\'s r': [round(np.std(delta['Pearson\'s r']), 3)], 'MAE': [round(np.std(delta['MAE']), 3)], 'RMSE': [round(np.std(delta['RMSE']), 3)]})
    delta = delta.append(average)
    delta = delta.append(std)
    delta = delta.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
    delta.to_csv("{}_CV_LGBM_Delta_Scoring.csv".format(name))

    #Train model on the entire dataset
    full_data = pd.merge(data, data, how='cross')
    full_data["Fingerprint"] =  full_data.Fingerprint_x.combine(full_data.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
    full_data["Delta"] = full_data.Y_x - full_data.Y_y # calculate Delta values
    full_data.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True)
    model.fit(np.vstack(full_data.Fingerprint.to_numpy()),full_data.Delta) 

    # Save the model to be referenced later
    filename = 'LGBM_delta_{}.sav'.format(name)
    pickle.dump(model, open(filename, 'wb'))
    
########################


##########################
### External Test Sets ###
##########################

# Dataset loading
properties = ['Sol', 'VDss']
loaded_model = pickle.load(open('LGBM_delta_{}.sav'.format(name), 'rb'))

for property in properties:
    dataset = '{}.csv'.format(property)
    pred_dataset = '{}ExternalTestSetFiltered.csv'.format(property)

    # Prepare data
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Molecule"] = mols
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})

    # Predictions
    pair = pd.merge(data, data, how='cross') # Cross merge the data together
    pair["Fingerprint"] =  pair.Fingerprint_x.combine(pair.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
    pair["Delta"] = pair.Y_x - pair.Y_y # calculate Delta values
    pair.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True) # rid of unnecessary files
    pair["Delta_pred"] = loaded_model.predict(np.vstack(pair.Fingerprint.to_numpy()))  # make predictions
    pair.drop(['Fingerprint'], axis=1, inplace=True) # rid of unnecessary files
    pair.to_csv('{}_LGBM_Delta_Preds_Ext.csv'.format(property), index = False) # save
    
##########################