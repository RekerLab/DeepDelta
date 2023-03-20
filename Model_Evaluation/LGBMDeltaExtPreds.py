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

# Dataset loading
properties = ['Sol', 'VDss']
loaded_model = pickle.load(open('LGBM_delta_{}.sav'.format(name), 'rb'))

for property in properties:
    dataset = '{}.csv'.format(property)
    pred_dataset = '{}ExternalTestSetFiltered.csv'.format(property)

    # Process data
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Molecule"] = mols
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  dataframe.Y.to_numpy()})

    train_x = dataframe[dataframe.columns[0]]
    dataset = pd.merge(train_x, train_x, how='cross')
    dataset = dataset[['SMILES_y', 'SMILES_x']]
    dataset = dataset.rename(columns={"SMILES_y": "SMILES_x", "SMILES_x": "SMILES_y"})

    pair = pd.merge(data, data, how='cross') # Cross merge the data together
    pair["Fingerprint"] =  pair.Fingerprint_x.combine(pair.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
    pair["Delta"] = pair.Y_x - pair.Y_y # calculate Delta values
    pair.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True) # rid of unnecessary files
    pair["Delta_pred"] = loaded_model.predict(np.vstack(pair.Fingerprint.to_numpy()))  # make predictions
    pair.drop(['Fingerprint'], axis=1, inplace=True) # rid of unnecessary files
    pair.to_csv('{}_LGBM_Delta_Preds_Ext.csv'.format(property), index = False) # save
