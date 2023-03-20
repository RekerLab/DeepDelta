#Import Libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold 
import math
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import metrics
from scipy import stats as stats

# Dataset loading:
names = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

for name in names:
    dataframe = pd.read_csv("{}.csv".format(name))

    # Prepare fingerprints
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'FP':  dataframe.Fingerprint.to_numpy()})
    del dataframe

    # Set up for cross validation
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # Use garbage collection to save on space
    import gc
    print(gc.get_count())
    gc.collect()
    print(gc.get_count())

    # Perform cross validation and save fingerprints
    FA = []
    FB = []

    for train_index, test_index in cv.split(data):

    train_df = data[data.index.isin(train_index)]
    test_df = data[data.index.isin(test_index)]
    pair_subset_test = pd.merge(test_df, test_df, how='cross')
    FA += [pair_subset_test.FP_x]
    FB += [pair_subset_test.FP_y]
    del pair_subset_test

    # Calculate the similarity values
    similarity_list = []
    for i in range(len(np.concatenate(FA))):
    similarity_list.append(DataStructs.TanimotoSimilarity(DataStructs.cDataStructs.CreateFromBitString("".join(np.concatenate(FA)[i].astype(str))), DataStructs.cDataStructs.CreateFromBitString("".join(np.concatenate(FB)[i].astype(str)))))

    #Export the csv containing similarity values
    dataframe = pd.DataFrame(similarity_list)
    dataframe.rename(columns={0: 'Tanimoto'}, inplace = True)
    dataframe.to_csv('{}_CV_Similarity_Scores.csv'.format(name), index = False)