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
  from sklearn.ensemble import RandomForestRegressor as RF
  model = RF()

  # Train model on the entire dataset
  model.fit(np.vstack(data.Fingerprint.to_numpy()),data.Y) 

  # Save the model to be referenced later
  filename = 'RF_control_{}.sav'.format(name)
  pickle.dump(model, open(filename, 'wb'))


  #### Modeling - LightGBM ####
  from lightgbm import LGBMRegressor as lgb
  model = lgb(subsample=0.1, subsample_freq = 1)

  # Train model on the entire dataset
  model.fit(np.vstack(data.Fingerprint.to_numpy()),data.Y) 

  # Save the model to be referenced later
  filename = 'LGBM_control_subsample_{}.sav'.format(name)
  pickle.dump(model, open(filename, 'wb'))
  