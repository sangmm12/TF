import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load models
models = []
for i in range(1, 3):  # Loading only model_1 and model_2
    with open(f'best_xgb_mouse_M36_model_{i}.pkl', 'rb') as file:
    #with open(f'best_models/best_xgb_human_H47_model_{i}.pkl', 'rb') as file:    
    #with open(f'best_models/best_xgb_mouse_M36_model_{i}.pkl', 'rb') as file:
        models.append(pickle.load(file))

# Load Gene File
#data = pd.read_csv('H47_NV_1368.csv')  # GeneName, Length, Seq, V1, ..., V1368
data = pd.read_csv('M36_NV_1368.csv')

# Prepare to store results
results = []

# Iterate over each gene
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    gene_name = row['GeneName']
    gene_features = row.iloc[3:].values.reshape(1, -1)  # Extracting V1, ..., V1368
    #print(f"Gene: {gene_name}, Features: {gene_features}")

    if hasattr(models[0], "predict_proba"):
        tf_prob = models[0].predict_proba(gene_features)[0][1]
        betf_prob = models[1].predict_proba(gene_features)[0][1]
    else:
        tf_prob = models[0].predict(gene_features)
        betf_prob = models[1].predict(gene_features)
    
    # Step 2: Predict if the gene_name is TF or BeTF using model_1 and model_2
    #tf_prob = models[0].predict_proba(gene_features)[0][1]
    #betf_prob = models[1].predict_proba(gene_features)[0][1]

    # Append results
    results.append([gene_name, tf_prob, betf_prob])

# Save the results to a file
results_df = pd.DataFrame(results, columns=['GeneName', 'TF Probability', 'RG Probability'])
#results_df.to_csv('HH_H47_TF.csv', index=False)  # Save to CSV file
results_df.to_csv('HH_M36_TF.csv', index=False) 

# Calculate counts based on the threshold
threshold = 0.8
tf_count = results_df[results_df['TF Probability'] > threshold].shape[0]
betf_count = results_df[results_df['RG Probability'] > threshold].shape[0]
both_count = results_df[(results_df['TF Probability'] > threshold) & (results_df['RG Probability'] > threshold)].shape[0]

print(f'Number of TF Probability > {threshold}: {tf_count}')
print(f'Number of RG Probability > {threshold}: {betf_count}')
print(f'Number of both TF Probability > {threshold} & RG Probability > {threshold}: {both_count}')

