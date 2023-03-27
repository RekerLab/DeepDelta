import pandas as pd
import numpy as np
import math


datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']
scoring = pd.DataFrame(columns=['Dataset', 'MAE1', 'RMSE1', 'MAE2', 'RMSE2', 'COR2', 'MAE3', 'RMSE3'])

def MAE(x, n):
  return np.sum(np.abs(x)) / n

def RMSE(x, n):
  return np.sqrt(np.sum(np.square(x)) / n)


for dataset in datasets:

  n = 0
  diagonal = []
  reverse_paired = []
  triangle_paired = []

  for i in range(10):
    fold = i+1
    preds = pd.read_csv('Results/DeepDelta_Individual_CV_Folds/{}_DeepDelta_Individual_Fold_{}.csv'.format(dataset, fold)).T

    n_fold = int(np.sqrt(len(preds)))
    matrix = np.reshape(np.array(preds), (n_fold, n_fold))
    n += n_fold
   
    diagonal = np.append(np.array(diagonal), np.diag(matrix)) # For Eq1
    reverse_paired = np.append(np.array(reverse_paired), (matrix + np.transpose(matrix)).flatten()) # For Eq2

    triangle_paired_fold = []
    for i in range(n_fold):
      for ii in range(n_fold):
        triangle_paired_fold.extend(matrix[i,:] + matrix[:,ii] - matrix[i,ii])   
    triangle_paired =  np.append(np.array(triangle_paired), triangle_paired_fold) # For Eq3

  ####################
  # Eq1: DD(x,x) = 0 #
  #################### 

  MAE1 = MAE(diagonal, n)
  RMSE1 = RMSE(diagonal, n)
  
  ####################

  ###########################
  # Eq2: DD(x,y) = -DD(y,x) #
  ###########################

  MAE2 = MAE(reverse_paired, n*n)
  RMSE2 = RMSE(reverse_paired, n*n)
  COR2 = np.corrcoef(matrix.flatten(), np.transpose(matrix).flatten())[0,1]

  ###########################


  ####################################
  # Eq3: DD(x,y) + DD(y,z) = DD(x,z) #
  ####################################
    
  MAE3 = MAE(triangle_paired, n**3)
  RMSE3 = RMSE(triangle_paired, n**3)
    
  ####################################

  scores = pd.DataFrame({'Dataset': [dataset], 'MAE1': [round(MAE1, 4)], 'RMSE1': [round(RMSE1, 4)], 
                    'MAE2': [round(MAE2, 4)], 'RMSE2': [round(RMSE2, 4)], 'COR2': [round(COR2, 4)],
                    'MAE3': [round(MAE3, 4)], 'RMSE3': [round(RMSE3, 4)]})
  scoring = pd.concat([scoring, scores])
scoring.to_csv("Mathematical_Invariants_Results.csv", index = False)




