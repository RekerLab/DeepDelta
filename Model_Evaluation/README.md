# Code for Model Evaluation

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN

## Installation
ChemProp can be be installed at https://github.com/chemprop/chemprop 

## DeepDelta5x10CV_ExtPred.py
Code to run 5x10 fold cross validation and external test set predictions for the DeepDelta model. Also contains functions for reverse predictions for Equation 2: DeepDelta(x,y)= -DeepDelta(y,x).

## ChemProp5x10CV_ExtPred.py
Code to run 5x10 fold cross validation and external test set predictions for the ChemProp model. 

## RF-LGBM-Control5x10CV.ipynb
Code to run 5x10 fold cross validation and external test set predictions for the Random Forest and LightGBM models. 

## RF-LGBM-CTRLExtPreds.ipynb

## LGBMDelta5x10CV.ipynb

## LGBMDeltaExtPreds.ipynb

## GenerateCTRLModels.ipynb

## PlotCVData.ipynb

## ChemSimilarityVsPredictions.ipynb

## SameScaffoldAnalysis.ipynb


## EquationOne.ipynb

## EquationThree.py
## EquationOne.ipynb
