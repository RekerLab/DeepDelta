# DeepDelta
DeepDelta is a pairwise deep learning approach that processes two molecules simultaneously and learns to predict property differences between two molecules.

![image](https://user-images.githubusercontent.com/127516906/224864795-0bb1e827-9447-489b-9a9b-a60f01bb2526.png)


**Figure 1: Traditional and Pairwise Architectures. (A)** Traditional molecular machine learning models take singular molecular inputs and predict absolute properties of molecules. Predicted property differences can be calculated by subtracting predicted values for two molecules. **(B)** Pairwise models train on differences in properties from pairs of molecules to directly predict property changes of molecular derivatizations. **(C)** Molecules cross-merged to create pairs only after cross-validation splits to prevent the risk of data leakage during model evaluation. Through this, every molecule in the dataset can only occur in pairs in the training or testing data but not both.
For more information, please refer to: 

If you use this data or code, please kindly cite:


# Descriptions of Folders

## Datasets

Training Data for 10 ADMET property benchmarking datasets and 2 external test sets.

## Model Evaluation

Python code for evaluating DeepDelta and traditional models.
