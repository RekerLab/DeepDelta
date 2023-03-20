# Code for Model Evaluation

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Comparison Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [ChemProp](https://github.com/chemprop/chemprop)
* [LightGBM](https://www.microsoft.com/en-us/research/project/lightgbm/)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN


## Installation
ChemProp can be installed from https://github.com/chemprop/chemprop 

## Python Scripts

#### DeepDelta5x10CV_ExtPred.py
Code to run 5x10 fold cross validation and external test set predictions for the DeepDelta model. 

Also contains function for reverse predictions for Eq. 2 (with swapped input molecules, predictions should be the inverse of the original predictions):
```math
DeepDelta(x,y)= -DeepDelta(y,x).
```

#### ChemProp5x10CV_ExtPred.py
Code to run 5x10 fold cross validation and external test set predictions for the ChemProp model. 

#### RF-LGBM-Control-5x10CV-SaveModels-ExtPreds.py
Code to run 5x10 fold cross validation, generate and save trained models, and run external test set predictions for the Random Forest and LightGBM models. 

#### LGBMDelta5x10CV.py
Code to run 5x10 fold cross validation and save the LightGBM delta models. 

#### LGBMDeltaExtPreds.py
Code to run external test set predictions for the LightGBM delta models from saved models.  

#### ChemSimilarityVsPredictions.py
Code to compare the chemical similarity of pairs to the delta values and predictive error of the pairs. 

#### SameScaffoldAnalysis.py
Code to compare the chemical pairs with shared scaffolds to pairs who do not share scaffolds. 

#### EquationOne.py
Code to calculate property differences for exact same molecular inputs to satisfy Eq 1 (with same molecule for both inputs, predictions should be zero): 
```math
DeepDelta(x,x)= 0. 
```

#### EquationThree.py
Code to calculate if the predicted difference between three molecules are additive to satisfy Eq 3 (predicted difference between three molecules should be additive):
```math
DeepDelta(x,y) + DeepDelta(y,z)= DeepDelta(x,z)
```
