# Code for Model Evaluation

## Model Training, Cross-Validation, and External Test Set Predictions

#### DeepDelta-5x10CV-ExtPred.py
* Code for the DeepDelta models. 

#### ChemProp-5x10CV-ExtPred.py
* Code for the ChemProp models. 

#### RF-5x10CV-ExtPred.py
* Code for the Random Forest models. 

#### LGBM-Control-5x10CV-ExtPred.py
* Code for the LightGBM traditional models. 

#### LGBM-Delta-5x10CV-ExtPred.py
* Code for the LightGBM delta models. 

<br>

## Additional Evaluation Methods

#### Scaffold-Chemical-Similiarity-Analysis.py
* Code to analyze the chemical similarity of pairs and compare the chemical pairs with shared scaffolds to pairs who do not share scaffolds. 

#### Mathematical-Principle-Analysis.py
* Code to calculate property differences of same molecule pairs for Eq 1 (with same molecule for both inputs, predictions should be zero): 
```math
DeepDelta(x,x)= 0. 
```
&nbsp;&nbsp;and calculate reverse predictions for Eq. 2 (with swapped input molecules, predictions should be the inverse of the original predictions):
```math
DeepDelta(x,y)= -DeepDelta(y,x).
```

  and calculate predicted differences between three molecules for Eq. 3 (predicted difference between three molecules should be additive):
```math
DeepDelta(x,y) + DeepDelta(y,z)= DeepDelta(x,z)
```
