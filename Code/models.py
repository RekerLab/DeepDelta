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


class abstractDeltaModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass    




class DeepDelta(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name 
        
        train = pd.merge(x, x, how='cross') 
        y_values = pd.merge(y, y, how='cross')
        train["Y"] = y_values.Y_y - y_values.Y_x
        del y_values

        temp_datafile = tempfile.NamedTemporaryFile() 
        train.to_csv(temp_datafile.name, index=False)
        
        arguments = [ 
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name, 
            '--dataset_type', 'regression', 
            '--save_dir', self.dirpath,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1', 
            '--epochs', str(self.epochs),
            '--metric', metric,
            '--number_of_molecules', '2',
            '--aggregation', 'sum'
        ]
        
        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

        temp_datafile.close()


    def predict(self, x):

        dataset = pd.merge(x, x, how='cross')

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args)

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions
    
    def __str__(self):
        return "DeepDelta" + str(self.epochs)




class Trad_ChemProp(abstractDeltaModel):
    epochs = None
    dirpath = None  
    dirpath_single = None

    def __init__(self, epochs=50, dirpath = None, dirpath_single = None):
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, y, metric='r2'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name
        
        train = pd.DataFrame(np.transpose(np.vstack([x,y])),columns=["X","Y"])

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'regression',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '1',
            '--metric', metric, 
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

        temp_datafile.close()


    def predict(self, x): 

        dataset = pd.DataFrame(x)
        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args)

        predictions = pd.read_csv(temp_predfile.name)['Y'] 

        preds = pd.merge(predictions,predictions,how='cross')

        temp_datafile.close()
        temp_predfile.close()

        return preds.Y_y - preds.Y_x 
    
    def __str__(self):
        return "ChemProp" + str(self.epochs)




class Trad_RF(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = RF()

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        self.model.fit(fps,y)

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps))
        results = pd.merge(predictions,predictions,how='cross')
        return results['0_y'] - results['0_x'] 
    
    def __str__(self):
        return "RandomForest"





class Trad_LGBM(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = lgb(subsample=0.1, subsample_freq = 1)

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        self.model.fit(fps,y)

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps))
        results = pd.merge(predictions,predictions,how='cross')
        return results['0_y'] - results['0_x'] 
    
    def __str__(self):
        return "LGBM"
        



class Delta_LGBM(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = lgb(subsample=0.1, subsample_freq = 1)

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  y})
        pair_data = pd.merge(data, data, how='cross')
        
        pair_data["Fingerprint"] =  pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append) 
        pair_data["Delta"] = pair_data.Y_x - pair_data.Y_y 
        
        pair_data.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True)
        self.model.fit(np.vstack(pair_data.Fingerprint.to_numpy()),pair_data.Delta) 
        del pair_data

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool))})
        pair_data = pd.merge(data, data, how='cross')
        pair_data["Fingerprint"] =  pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append) 
        pair_data.drop(['Fingerprint_x','Fingerprint_y'], axis=1, inplace=True)
        
        predictions = pd.DataFrame(self.model.predict(pair_data))
        return predictions
    
    def __str__(self):
        return "DeltaLGBM"

