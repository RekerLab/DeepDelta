import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold 
import math
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import metrics
from scipy import stats as stats
import pickle

# Load datasets
properties = ['Sol', 'VDss']
loaded_RF_model = pickle.load(open('RF_control_{}.sav'.format(name), 'rb'))
loaded_LGBM_model = pickle.load(open('LGBM_control_subsample_{}.sav'.format(name), 'rb'))

for property in properties:
    dataset = '{}.csv'.format(property)
    pred_dataset = '{}ExternalTestSetFiltered.csv'.format(property)')

    # process data
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Molecule"] = mols
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})

    # Make predictions - Random Forest
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

