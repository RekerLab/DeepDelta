import pandas as pd
import numpy as np

properties = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

for property in properties:
    dataset = '{}.csv'.format(property)
    results_df = pd.DataFrame(columns=['Fold', 'Average', 'Std Dev'])
    overall_error = pd.DataFrame(columns=['Error'])

    for a in range(10):
      cnt = a+1
      full_data = pd.read_csv("{}_DeepDelta_EachCVFold_All_{}.csv".format(name, cnt))
      df = pd.DataFrame(index=range(len(full_data['SMILES_x'].unique())),columns=range(len(full_data['SMILES_x'].unique())))

      # Matrix
      for i in range(len(full_data['SMILES_x'].unique())):
        for j in range(len(full_data['SMILES_x'].unique())):
          position = i*len(full_data['SMILES_x'].unique()) + j
          df.iat[i, j] =  full_data['Delta_preds'][position]

      #Calculations 
      new_df = pd.DataFrame(columns = ['Error'])

      #Traverse the matrix making all the calculations
      for i in range(len(df[0])): # Main column
        for j in range(len(df[0])): # Main row
          for k in range(len(df[0])): # Secondary column/row movement
            error = round(abs(df.iat[i,j] - (df.iat[i,k] + df.iat[k,j])), 4)
            inter_df = pd.DataFrame({'Error': [error]})
            new_df = pd.concat([new_df, inter_df])
            overall_error = pd.concat([overall_error, inter_df])

      int_df = pd.DataFrame({'Fold': [a], 'Average': [round(new_df.mean(), 4)], 'Std Dev': [round(new_df.std(), 4)]})
      results_df = results_df.append(int_df)
      print("{}/10 done.".format(cnt))

    overall_df = pd.DataFrame({'Fold': ['Overall'], 'Average': [round(overall_error.mean(), 4)], 'Std Dev': [round(overall_error.std(), 4)]})
    results_df = results_df.append(overall_df)

    pd.DataFrame(results_df).to_csv("{}_DeepDelta_Triangle_Inequality_Error.csv".format(property), index=False)