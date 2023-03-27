import os
import tempfile
import shutil
import abc
import pandas as pd
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
from lightgbm import LGBMRegressor as lgb


properties = ['Sol', 'VDss']
models = [DeepDelta(), Trad_ChemProp(), Trad_RF(), Trad_LGBM(), Delta_LGBM()]

for model in models:
    for prop in properties:
        dataset = 'Datasets/Benchmarks/{}.csv'.format(prop)
        pred_dataset = 'Datasets/External/{}ExternalTestSetFiltered.csv'.format(prop)
        
        df = pd.read_csv(dataset)
        x = df[df.columns[0]]
        y = df[df.columns[1]]
        
        model.fit(x,y)
        
        df = pd.read_csv(pred_dataset)
        x = df[df.columns[0]]
        y = df[df.columns[1]]
        
        y_pairs = pd.merge(y, y, how='cross')
        vals = y_pairs.Y_y - y_pairs.Y_x
        preds = model.predict(x)
        
        results = [vals, preds]
        pd.DataFrame(results).to_csv("{}_{}_ExtTestSet.csv".format(prop, model), index=False)
        #If you .T the dataframe, then the first column is ground truth, the second is predictions

