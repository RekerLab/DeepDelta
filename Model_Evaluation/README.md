# Code for Model Evaluation

#### DeepDelta5x10CV_ExtPred.py
Code to run 5x10 fold cross validation and external test set predictions for the DeepDelta model. 

#### ChemProp5x10CV_ExtPred.py
Code to run 5x10 fold cross validation and external test set predictions for the ChemProp model. 

#### RF-LGBM-Control-5x10CV-ExtPreds.py
Code to run 5x10 fold cross validation, save trained models, and run external test set predictions for the Random Forest and LightGBM models. 

#### LGBM-Delta-5x10CV-ExtPreds.py
Code to run 5x10 fold cross validation, save trained models, and run external test set predictions for the LightGBM delta models. 

#### Scaffold-Chemical-Similiarity-Analysis.py
Code to analyze the chemical similarity of pairs and compare the chemical pairs with shared scaffolds to pairs who do not share scaffolds. 

#### Mathematical-Principle-Analysis.py
Code to calculate property differences of same molecule pairs for Eq 1 (with same molecule for both inputs, predictions should be zero): 
```math
DeepDelta(x,x)= 0. 
```
and calculate reverse predictions for Eq. 2 (with swapped input molecules, predictions should be the inverse of the original predictions):
```math
DeepDelta(x,y)= -DeepDelta(y,x).
```

and calculate predicted differences between three molecules for Eq. 3 (predicted difference between three molecules should be additive):
```math
DeepDelta(x,y) + DeepDelta(y,z)= DeepDelta(x,z)
```
