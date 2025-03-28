import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# Step 1: Load models
hm = 'human'#'mouse'#'human'  # Change this if needed ('mouse' or 'human')
version = 'H47'#'M36'#'H47'  # Change this if needed ('M36' or 'H47')
models = []

# Load models 3 to 6
for i in range(3, 7):  
    #with open(f'{hm}/best_xgb_{hm}_{version}_model_{i}.pkl', 'rb') as file:
    with open(f'best_xgb_{hm}_{version}_model_{i}.pkl', 'rb') as file:
        models.append(pickle.load(file))

# Load data
# data_seq = pd.read_csv(f'{version}_NV_1368.csv')  # Example for Human data
#data_seq = pd.read_csv('M36_M30_M24_NV_1368.csv')  # for evaluating models 3/4/5/6
data_seq = pd.read_csv('H47_H43_H31_NV_1368.csv')  # for evaluating models 3/4/5/6

# data_TF = pd.read_csv(f'predict_results/{version}_TF.csv')  # Example for Human data
#data_TF = pd.read_csv(f'HH_{version}_TF.csv') #not used in this code

# Step 2: Build TF_Set and RG_Set from original dataset for evaluating model 3/4/5/6
df = pd.read_csv(f'TF_{hm}.csv')  # id, source, target, mor

# Extract unique pairs from the 'source' and 'target' columns
TF_RG_pairs = [(source.lower(), target.lower()) for source, target in zip(df['source'], df['target'])]
RG_TF_pairs = [(target.lower(), source.lower()) for target, source in zip(df['target'], df['source'])]
print(f"Count of TF-RG Pairs: {len(TF_RG_pairs)}")
print(f"Count of RG-TF Pairs: {len(RG_TF_pairs)}")

# Count unique sources and targets (case-insensitive)
unique_sources = df['source'].str.lower().unique()
unique_targets = df['target'].str.lower().unique()
print(f"Count of unique sources: {len(unique_sources)}")
print(f"Count of unique targets: {len(unique_targets)}")

# Define gene set (same as before)
human_set = {
    'ID1', 'ZBTB18', 'PAX6', 'SOX4', 'FEZF2', 'PBX2',
    'LHX2', 'EMX2', 'SOX11', 'TCF4', 'POU3F2', 'SOX3',
    'MYC', 'CCND1', 'HIF1A', 'EPO', 'NFKB1', 'TNF',
    'STAT3', 'BCL2', 'ESR1', 'GREB1', 'FOXO3', 'GADD45A',
    'GATA3', 'IL4', 'PDX1', 'SOX9', 'COL2A1',
    'POU5F1', 'FGF4', 'RUNX2', 'SP7', 'SMAD4', 'CDH1',
    'TP53', 'RB1', 'FOXO1', 'GATA1', 'STAT5', 'E2F1',
    'HES1', 'TGFB1', 'ZEB1', 'NFKBIA', 'CDKN1A', 'MYB', 'CDKN2B', 'MDM2', 'MYOD1'
}
mouse_set = {
    'Id1', 'Zbtb18', 'Pax6', 'Sox4', 'Fezf2', 'Pbx2',
    'Lhx2', 'Emx2', 'Sox11', 'Tcf4', 'Pou3f2', 'Sox3',
    'Myc', 'Ccnd1', 'Hif1a', 'Epo', 'Nfkb1', 'Tnf',
    'Stat3', 'Bcl2', 'Esr1', 'Greb1', 'Foxo3', 'Gadd45a',
    'Gata3', 'Il4', 'Pdx1', 'Sox9', 'Col2a1',
    'Pou5f1', 'Fgf4', 'Runx2', 'Sp7', 'Smad4', 'Cdh1',
    'Tp53', 'Rb1', 'Foxo1', 'Gata1', 'Stat5', 'E2f1',
    'Hes1', 'Tgf-β', 'Zeb1', 'Nfkbia', 'CdkN1a', 'Myb', 'Cdk2b', 'Mdm2', 'Myod1'
}

gene_set = human_set  # You can switch to mouse_set if needed
using_sub_gen_set = False  # Set this to True if using a subset of the gene set

# Step 3: Determine links between TF and RG genes (Handle only the pairs in TF_RG_pairs)
#link_threshold = 0.5  # Threshold for link probability
#activation_threshold = 0.1  # Threshold for activation probability
#repression_threshold = 0.1  # Threshold for repression probability

#for all possible links and activation/repression
link_threshold = 0.0  # Threshold for link probability
activation_threshold = 0.0  # Threshold for activation probability
repression_threshold = 0.0  # Threshold for repression probability

# Initialize counts
model_3_count = 0
model_4_count = 0
model_5_count = 0
model_6_count = 0

link_results = []
notfind_count = 0

# Using model_3 for TF -> RG links
for tf_gene, rg_gene in tqdm(TF_RG_pairs, desc="Processing TF -> RG links"):
    if using_sub_gen_set and (tf_gene not in gene_set or rg_gene not in gene_set):
        continue
    if tf_gene != rg_gene:
        # Check if tf_gene and rg_gene exist in data_seq
        if tf_gene in data_seq['GeneName'].str.lower().values and rg_gene in data_seq['GeneName'].str.lower().values:
            tf_features = data_seq[data_seq['GeneName'].str.lower() == tf_gene].iloc[0, 3:].values
            rg_features = data_seq[data_seq['GeneName'].str.lower() == rg_gene].iloc[0, 3:].values
            pair_features = np.concatenate((tf_features, rg_features)).reshape(1, -1)
            link_prob = models[0].predict_proba(pair_features)[0][1]
            if link_prob > link_threshold:
                link_results.append((tf_gene, rg_gene, link_prob, 'model_3'))
                model_3_count += 1
        else:
            notfind_count += 1

# Using model_4 for RG -> TF links
for rg_gene, tf_gene in tqdm(RG_TF_pairs, desc="Processing RG -> TF links"):
    if using_sub_gen_set and (rg_gene not in gene_set or tf_gene not in gene_set):
        continue
    if rg_gene != tf_gene:
        # Check if rg_gene and tf_gene exist in data_seq
        if rg_gene in data_seq['GeneName'].str.lower().values and tf_gene in data_seq['GeneName'].str.lower().values:
            rg_features = data_seq[data_seq['GeneName'].str.lower() == rg_gene].iloc[0, 3:].values
            tf_features = data_seq[data_seq['GeneName'].str.lower() == tf_gene].iloc[0, 3:].values
            pair_features = np.concatenate((tf_features, rg_features)).reshape(1, -1)
            link_prob = models[1].predict_proba(pair_features)[0][1]
            if link_prob > link_threshold:
                link_results.append((tf_gene, rg_gene, link_prob, 'model_4'))
                model_4_count += 1
        else:
            notfind_count += 1

print(f"================================================================notfind_count: {notfind_count}")

# Output the initial length of link_results
total_link = len(link_results)
print(f"Initial length of link_results: {total_link}")

# Further analysis for final_results
final_results = []
for idx, (tf_gene, rg_gene, link_prob, model_type) in enumerate(link_results):
    tf_features = data_seq[data_seq['GeneName'].str.lower() == tf_gene].iloc[0, 3:].values
    rg_features = data_seq[data_seq['GeneName'].str.lower() == rg_gene].iloc[0, 3:].values
    pair_features = np.concatenate((tf_features, rg_features)).reshape(1, -1)

    if idx % 10000 == 0:
        print(f"Index: {idx}/{total_link}, TF Gene: {tf_gene}, RG Gene: {rg_gene}")

    activation_prob = models[2].predict_proba(pair_features)[0][1]
    repression_prob = models[3].predict_proba(pair_features)[0][1]

    if activation_prob > activation_threshold:
        final_results.append((tf_gene, rg_gene, activation_prob, 'model_5'))
        model_5_count += 1
    
    if repression_prob > repression_threshold:
        final_results.append((tf_gene, rg_gene, repression_prob, 'model_6'))
        model_6_count += 1

# Combine link_results and final_results
combined_results = link_results + final_results

# Save results in a readable format, removing duplicate records
results_df = pd.DataFrame(combined_results, columns=['TF_Gene', 'RG_Gene', 'Link_Probability', 'Model_Type'])
results_df.drop_duplicates(inplace=True)
results_df.to_csv(f"model3456_{version}_{hm}_set_evaluating_ALL.csv", index=False)

# Output counts for each model
print(f"Model_3 Count: {model_3_count}")
print(f"Model_4 Count: {model_4_count}")
print(f"Model_5 Count: {model_5_count}")
print(f"Model_6 Count: {model_6_count}")

