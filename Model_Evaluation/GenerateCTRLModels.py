import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle 
from sklearn.ensemble import RandomForestRegressor as RF
from lightgbm import LGBMRegressor as lgb

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

  #### Modeling - Random Forest ####
  model = RF()
  model.fit(np.vstack(data.Fingerprint.to_numpy()),data.Y) # Train model on the entire dataset
  filename = 'RF_control_{}.sav'.format(name)
  pickle.dump(model, open(filename, 'wb')) # Save the model to be referenced later


  #### Modeling - LightGBM ####
  if dataset == FUBrain:
    model = lgb(subsample=0.1, subsample_freq = 1, min_child_samples = 5) # Only for FUBrain dataset due to small size
  else:
    model = lgb(subsample=0.1, subsample_freq = 1)
  model.fit(np.vstack(data.Fingerprint.to_numpy()),data.Y) # Train model on the entire dataset
  filename = 'LGBM_control_subsample_{}.sav'.format(name)
  pickle.dump(model, open(filename, 'wb')) # Save the model to be referenced later
  