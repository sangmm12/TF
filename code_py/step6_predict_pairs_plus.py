import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# Step 1: Load models
hm = 'mouse'#human'
version = 'M36'#'H47'
models = []
for i in range(3, 7):  # Load models 3 to 6
    #with open(f'{hm}/best_xgb_{hm}_{version}_model_{i}.pkl', 'rb') as file:
    with open(f'best_xgb_{hm}_{version}_model_{i}.pkl', 'rb') as file:
        models.append(pickle.load(file))

# Load data
#data_seq = pd.read_csv(f'{version}_NV_1368.csv')  # GeneName, Length, Seq, V1, ..., V1368 (Human Data Example)
#data_seq = pd.read_csv('H47_H43_H31_NV_1368.csv') # for evaluating model 3/4/5/6 since some gene is not in H47_NV_1368.csv, hope this file would help
data_seq = pd.read_csv('M36_M30_M24_NV_1368.csv')

#data_TF = pd.read_csv(f'predict_results/{version}_TF.csv')  # GeneName, TF Probability, RG Probability (Human Data Example)
data_TF = pd.read_csv(f'HH_{version}_TF.csv') 

# Step 2: Build TF_Set and RG_Set
TF_Threshold = 0.9995
RG_Threshold = 0.9995
TF_Set_ = set(data_TF[data_TF['TF Probability'] > TF_Threshold]['GeneName'])
RG_Set_ = set(data_TF[data_TF['RG Probability'] > RG_Threshold]['GeneName'])
print(f"Count of TF_Set: {len(TF_Set_)}")
print(f"Count of RG_Set: {len(RG_Set_)}")

# 将集合转换为 DataFrame 并保存为 CSV
pd.DataFrame(list(TF_Set_), columns=['GeneName']).to_csv(f'{version}_TF_Set_999.csv', index=False)
pd.DataFrame(list(RG_Set_), columns=['GeneName']).to_csv(f'{version}_RG_Set_999.csv', index=False)

'''
# 定义要过滤掉的前缀
prefixes_to_filter_out = ('ensg', 'ensmusg', 'gm')
prefixes_to_filter_out_lower = [prefix.lower() for prefix in prefixes_to_filter_out]  # 转换为小写列表
# 过滤 TF_Set，去除具有指定前缀的基因
TF_Set = {gene for gene in TF_Set_ if not any(gene.lower().startswith(prefix) for prefix in prefixes_to_filter_out_lower)}
# 过滤 RG_Set，去除具有指定前缀的基因
RG_Set = {gene for gene in RG_Set_ if not any(gene.lower().startswith(prefix) for prefix in prefixes_to_filter_out_lower)}
print(f"Filtered Count of TF_Set (excluding prefixes): {len(TF_Set)}")
print(f"Filtered Count of RG_Set (excluding prefixes): {len(RG_Set)}")

'''
# Build TF_Set/RG_Set from original dataset for evaluating model 3/4/5/6
df = pd.read_csv('TF_human.csv') #id,source,target,mor
# Extract unique values for 'source' and 'target' columns
TF_Set = [source.lower() for source in df['source'].unique()]
RG_Set = [target.lower() for target in df['target'].unique()]
print(f"Count of TF_Set: {len(TF_Set)}")
print(f"Count of RG_Set: {len(RG_Set)}")
'''

# Define gene set
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

#gene_set = {'Id1', 'Zbtb18', 'Pax6', 'Sox4', 'Fezf2', 'Pbx2', 'Lhx2', 'Emx2', 'Sox11', 'Tcf4', 'Pou3f2', 'Sox3'}
#gene_set = {'Aatf', 'Abl1', 'Aebp2', 'Aes', 'Ahr', 'Bak1', 'Bax', 'Bbc3', 'Cdkn1a', 'Tpt1', 'Trp53', 'Trp73', 'Jarid2', 'Snai2', 'Ihh', 'Abcc4', 'Apaf1', 'Cyp19a1', 'Cyp1a1'}
gene_set = human_set#mouse_set
using_sub_gen_set = False#True

# Step 3: Determine links between TF and RG genes
link_threshold = 0.0#0.5
activation_threshold = 0.0#0.5
repression_threshold = 0.0#0.5

# Initialize counts
model_3_count = 0
model_4_count = 0
model_5_count = 0
model_6_count = 0

link_results = []
notfind_count = 0
# Using model_3 for TF -> RG links
for tf_gene in tqdm(TF_Set, desc="Processing TF -> RG links"):
    for rg_gene in tqdm(RG_Set, desc=f"Processing links for {tf_gene}", leave=False):
        if using_sub_gen_set and (tf_gene.lower() not in [g.lower() for g in gene_set] or rg_gene.lower() not in [g.lower() for g in gene_set]):
            continue
        if tf_gene != rg_gene:
            # Check if tf_gene and rg_gene exist in data_seq
            if tf_gene.lower() in data_seq['GeneName'].str.lower().values and rg_gene.lower() in data_seq['GeneName'].str.lower().values:
                #print(rg_gene)
                tf_features = data_seq[data_seq['GeneName'].str.lower() == tf_gene.lower()].iloc[0, 3:].values
                rg_features = data_seq[data_seq['GeneName'].str.lower() == rg_gene.lower()].iloc[0, 3:].values
                pair_features = np.concatenate((tf_features, rg_features)).reshape(1, -1)
                link_prob = models[0].predict_proba(pair_features)[0][1]
                if link_prob > link_threshold:
                    link_results.append((tf_gene, rg_gene, link_prob, 'model_3'))
                    model_3_count += 1
            else:
                notfind_count += 1

# Using model_4 for RG -> TF links
for rg_gene in tqdm(RG_Set, desc="Processing RG -> TF links"):
    for tf_gene in tqdm(TF_Set, desc=f"Processing links for {rg_gene}", leave=False):
        if using_sub_gen_set and (rg_gene.lower() not in [g.lower() for g in gene_set] or tf_gene.lower() not in [g.lower() for g in gene_set]):
            continue
        if rg_gene != tf_gene:
            # Check if rg_gene and tf_gene exist in data_seq
            if rg_gene.lower() in data_seq['GeneName'].str.lower().values and tf_gene.lower() in data_seq['GeneName'].str.lower().values:
                rg_features = data_seq[data_seq['GeneName'].str.lower() == rg_gene.lower()].iloc[0, 3:].values
                tf_features = data_seq[data_seq['GeneName'].str.lower() == tf_gene.lower()].iloc[0, 3:].values
                pair_features = np.concatenate((tf_features, rg_features)).reshape(1, -1)
                link_prob = models[1].predict_proba(pair_features)[0][1]
                if link_prob > link_threshold:
                    link_results.append((tf_gene, rg_gene, link_prob, 'model_4'))
                    model_4_count += 1
            else:
                notfind_count += 1

print(f"================================================================\nnotfind_count: {notfind_count}")

# Output the initial length of link_results
total_link = len(link_results)
print(f"Initial length of link_results: {total_link}")

# Further analysis for final_results
final_results = []
for idx, (tf_gene, rg_gene, link_prob, model_type) in enumerate(link_results):
    tf_features = data_seq[data_seq['GeneName'].str.lower() == tf_gene.lower()].iloc[0, 3:].values
    rg_features = data_seq[data_seq['GeneName'].str.lower() == rg_gene.lower()].iloc[0, 3:].values
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
results_df.to_csv(f"model3456_{version}_{hm}_set_forGNN_.csv", index=False)

# Output counts for each model
print(f"Model_3 Count: {model_3_count}")
print(f"Model_4 Count: {model_4_count}")
print(f"Model_5 Count: {model_5_count}")
print(f"Model_6 Count: {model_6_count}")
'''
