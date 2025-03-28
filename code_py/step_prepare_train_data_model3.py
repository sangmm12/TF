import pandas as pd
import numpy as np

hm = 'mouse'#'human'
model = 'model_5' #only for model 3/4
version = 'M36'#'H47'
# Load the dataset
file_path = f"{hm}/TF_{hm}_{version}_{model}_NV_2736.csv" #GeneName_1,GeneName_2,Interaction,Seq_1,Seq_2,V1,...,V2736

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

# Split True samples into train, val, and test
train_true = true_samples.sample(frac=0.90, random_state=42)
val_true = true_samples.drop(train_true.index).sample(frac=0.5, random_state=42)
test_true = true_samples.drop(train_true.index).drop(val_true.index)

# Split False samples into train, val, and test
train_false = false_samples_balanced.sample(frac=0.90, random_state=42)
val_false = false_samples_balanced.drop(train_false.index).sample(frac=0.5, random_state=42)
test_false = false_samples_balanced.drop(train_false.index).drop(val_false.index)

# Combine the datasets
train_set = pd.concat([train_true, train_false])
val_set = pd.concat([val_true, val_false])
test_set = pd.concat([test_true, test_false])

# Assign Type labels
train_set['Type'] = 'train'
val_set['Type'] = 'val'
test_set['Type'] = 'test'

# Combine all sets back
final_data = pd.concat([train_set, val_set, test_set])

# Save the results to a new file
output_path = f"TF_{hm}_{version}_{model}_NV_2736_Type.csv"

final_data.to_csv(output_path, index=False)

