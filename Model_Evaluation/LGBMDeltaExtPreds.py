# Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

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
