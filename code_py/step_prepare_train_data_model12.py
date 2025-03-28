import pandas as pd
import numpy as np

# Load the dataset
#file_path = "human/TF_human_H47_model_1_NV_1368.csv"
#file_path = "human/TF_human_H47_model_2_NV_1368.csv"
#file_path = "mouse/TF_mouse_M36_model_1_NV_1368.csv"
file_path = "mouse/TF_mouse_M36_model_2_NV_1368.csv"

data = pd.read_csv(file_path)

# Add 'Type' column with initial value 'NONE'
data.insert(1, 'Type', 'NONE')

# Convert Interaction to boolean for easier filtering
data['Interaction'] = data['Interaction'].astype(bool)

# Separate True and False samples
true_samples = data[data['Interaction'] == True]
false_samples = data[data['Interaction'] == False]

# Randomly sample an equal number of False samples as True samples
false_samples_balanced = false_samples.sample(n=len(true_samples), random_state=42)

# Combine balanced dataset
balanced_data = pd.concat([true_samples, false_samples_balanced])

# Ensure reproducibility
np.random.seed(42)

# Split data into train, val, and test sets
#gene_set = {'sry', 'cebpe', 'rorc', 'stat2', 'trim21', 'znf492', 'fam170a', 'gata1', 'hdac3', 'creb3l4'}
gene_set = {'mt-nd3', 'trdv1', 'igkv2d-26', 'mc4r', 'sumo4', 'tnp1', 'ahsp', 'ccl1', 'defa6', 'cd3d'}

# Ensure the True samples containing gene_set are included in train
train_true = true_samples[true_samples['GeneName'].isin(gene_set)]
remaining_true = true_samples[~true_samples['GeneName'].isin(gene_set)]

# Split remaining true samples into train, val, and test
train_true_remaining = remaining_true.sample(frac=0.90, random_state=42)
val_true = remaining_true.drop(train_true_remaining.index).sample(frac=0.5, random_state=42)
test_true = remaining_true.drop(train_true_remaining.index).drop(val_true.index)

# Split False samples into train, val, and test
train_false = false_samples_balanced.sample(frac=0.90, random_state=42)
val_false = false_samples_balanced.drop(train_false.index).sample(frac=0.5, random_state=42)
test_false = false_samples_balanced.drop(train_false.index).drop(val_false.index)

# Combine the datasets
train_set = pd.concat([train_true, train_true_remaining, train_false])
val_set = pd.concat([val_true, val_false])
test_set = pd.concat([test_true, test_false])

# Assign Type labels
train_set['Type'] = 'train'
val_set['Type'] = 'val'
test_set['Type'] = 'test'

# Combine all sets back
final_data = pd.concat([train_set, val_set, test_set])

# Save the results to a new file
#output_path = "TF_human_H47_model_1_NV_1368_Type.csv"
#output_path = "TF_human_H47_model_2_NV_1368_Type.csv"
#output_path = "TF_mouse_M36_model_1_NV_1368_Type.csv"
output_path = "TF_mouse_M36_model_2_NV_1368_Type.csv"

final_data.to_csv(output_path, index=False)

