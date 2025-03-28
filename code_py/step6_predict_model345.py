import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Load models
hm = 'mouse'#human
version = 'M36'#H47
models = []
for i in range(3, 7):  # Load models 3 to 6
    with open(f'best_models/best_xgb_{hm}_{version}_model_{i}.pkl', 'rb') as file:
        models.append(pickle.load(file))

# Load data
data_seq = pd.read_csv(f'{version}_NV_1368.csv')  # GeneName, Length, Seq, V1, ..., V1368 (Human Data Example)

data_TF = pd.read_csv(f'{version}_TF.csv')  # GeneName, TF Probability, RG Probability (Human Data Example)

# Step 1: Build TF_Set and RG_Set
TF_Threshold = 0.99
RG_Threshold = 0.99

TF_Set_ = set(data_TF[data_TF['TF Probability'] > TF_Threshold]['GeneName'])
RG_Set_ = set(data_TF[data_TF['RG Probability'] > RG_Threshold]['GeneName'])
print(f"Count of TF_Set: {len(TF_Set_)}")
print(f"Count of RG_Set: {len(RG_Set_)}")

# 定义要过滤掉的前缀
prefixes_to_filter_out = ('ensg', 'ensmusg', 'gm')
prefixes_to_filter_out_lower = [prefix.lower() for prefix in prefixes_to_filter_out]  # 转换为小写列表
# 过滤 TF_Set，去除具有指定前缀的基因
TF_Set = {gene for gene in TF_Set_ if not any(gene.lower().startswith(prefix) for prefix in prefixes_to_filter_out_lower)}
# 过滤 RG_Set，去除具有指定前缀的基因
RG_Set = {gene for gene in RG_Set_ if not any(gene.lower().startswith(prefix) for prefix in prefixes_to_filter_out_lower)}
print(f"Filtered Count of TF_Set (excluding prefixes): {len(TF_Set)}")
print(f"Filtered Count of RG_Set (excluding prefixes): {len(RG_Set)}")

subset = {'Id1', 'Zbtb18', 'Pax6', 'Sox4', 'Fezf2', 'Pbx2', 'Lhx2', 'Emx2', 'Sox11', 'Tcf4', 'Pou3f2', 'Sox3'}

# Step 2: Build model_356_predict_matrix
def build_predict_matrix(row_set, col_set, model_indices, header_suffix):
    row_list = list(row_set)  # Convert to list
    col_list = list(col_set)  # Convert to list

    results = []  # Collect results in a list

    for row_gene in tqdm(row_list, desc="Building Matrix Rows"):
        # Debug: Only process if row_gene is in the subset (case insensitive)
        #if row_gene.lower() not in {gene.lower() for gene in subset}:
        #    continue
        #print(f"Processing row_gene: {row_gene}")  # Debug statement

        row_results = {'row_gene': row_gene}  # Initialize with the row_gene
        for col_gene in col_list:
            # Skip same gene comparisons
            if row_gene == col_gene:
                continue

            # Extract features (excluding 'Seq' column)
            row_features = data_seq[data_seq['GeneName'].str.lower() == row_gene.lower()].iloc[0, 3:].values
            col_features = data_seq[data_seq['GeneName'].str.lower() == col_gene.lower()].iloc[0, 3:].values

            pair_features = np.concatenate((row_features, col_features)).reshape(1, -1)

            # Predict probabilities for specified models
            probs = [models[idx].predict_proba(pair_features)[0][1] for idx in model_indices]

            # Collect probabilities
            for i, model_idx in enumerate(model_indices):
                row_results[f"{col_gene}_#{model_idx+3}"] = probs[i]

        results.append(row_results)

    # Convert results to DataFrame and set 'row_gene' as the index
    matrix = pd.DataFrame(results)
    matrix.set_index('row_gene', inplace=True)
    return matrix

# Step 3:
# Build model_356_predict_matrix (TF_Set as rows, RG_Set as columns)
model_356_predict_matrix = build_predict_matrix(TF_Set, RG_Set, [0, 2, 3], "#3,#5,#6")
model_356_predict_matrix.to_csv(f'{hm}_model_356_predict_matrix_99_test.csv', index=True)

# Build model_456_predict_matrix (RG_Set as rows, TF_Set as columns)
model_456_predict_matrix = build_predict_matrix(RG_Set, TF_Set, [1, 2, 3], "#4,#5,#6")
model_456_predict_matrix.to_csv(f'{hm}_model_456_predict_matrix_99_test.csv', index=True)

