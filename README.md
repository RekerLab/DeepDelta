

![DeepDelta7](https://user-images.githubusercontent.com/127516906/227276369-2af92e68-1e3d-436a-9d62-68567fbf2f7a.png)


## Overview

DeepDelta is a pairwise deep learning approach that processes two molecules simultaneously and learns to predict property differences between two molecules. 

![image](https://user-images.githubusercontent.com/127516906/225358174-ecb26783-a551-47c4-90f3-6950babee377.png)
**Figure 1: Traditional and Pairwise Architectures. (A)** Traditional molecular machine learning models take singular molecular inputs and predict absolute properties of molecules. Predicted property differences can be calculated by subtracting predicted values for two molecules. **(B)** Pairwise models train on differences in properties from pairs of molecules to directly predict property changes of molecular derivatizations. **(C)** Molecules are cross-merged to create pairs only after cross-validation splits to prevent the risk of data leakage during model evaluation. Through this, every molecule in the dataset can only occur in pairs in the training or testing data but not both.

On 10 pharmacokinetic benchmark tasks, our DeepDelta approach outperforms two established molecular machine learning algorithms, the message passing neural network (MPNN) ChemProp and Random Forest using radial fingerprints. 

We also derive three simple computational tests of our models based on first mathematical principles and show that compliance to these tests correlate with overall model performance – providing an innovative, unsupervised, and easily computable measure of expected model performance and applicability. 


<p align="center">
1. With same molecule for both inputs, predictions should be zero:
</p>


```math
DeepDelta(x,x)= 0
```

<br />

<p align="center">
2. With swapped input molecules, predictions should be inversed:
</p>


```math
DeepDelta(x,y)= - DeepDelta(y,x) 
```

<br />

<p align="center">
3. Predicted difference between three molecules should be additive:
</p>

 
```math
DeepDelta(x,y) + DeepDelta(y,z)= DeepDelta(x,z)
```

<br />


For more information, please refer to: *pre-print link*

If you use this data or code, please kindly cite: *pre-print citation*

<br />

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Comparison Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [ChemProp v1.5.2](https://github.com/chemprop/chemprop)
* [LightGBM](https://www.microsoft.com/en-us/research/project/lightgbm/)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Installation
ChemProp can be be installed from https://github.com/chemprop/chemprop 

<br />

## Descriptions of Folders

### Code

Python code for evaluating DeepDelta and traditional models based on their ability to predict property differences between two molecules.

### Datasets

Curated data for 10 ADMET property benchmarking training sets and 2 external test sets.

### Results

Results from 5x10-fold cross-validation that are utilized in further analysis.

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
