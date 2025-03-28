import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

hm = 'human'#'mouse'
model = 'model_5'
version = 'H47'#'M36'

# Load the dataset
file_path = f"{hm}/TF_{hm}_{version}_{model}_NV_2736.csv"  # GeneName_1, GeneName_2, Interaction, Seq_1, Seq_2, V1,..., V2736
data = pd.read_csv(file_path)

# Add 'Type' column with initial value 'NONE'
data.insert(1, 'Type', 'NONE')

# Convert Interaction to boolean for easier filtering
data['Interaction'] = data['Interaction'].astype(bool)

# Separate True and False samples
true_samples = data[data['Interaction'] == True]
false_samples = data[data['Interaction'] == False]

# Load the trained XGBoost model using pickle
model_path = f'best_xgb_{hm}_{version}_model_3.pkl'
with open(model_path, 'rb') as file:
    model_3 = pickle.load(file)

# Get the feature columns (V1 to V2736)
feature_columns = [f"V{i}" for i in range(1, 2737)]

# Prepare the feature data for the false samples
false_features = false_samples[feature_columns]

# Convert false_features to DMatrix to ensure GPU compatibility
false_features_gpu = xgb.DMatrix(false_features)  # Use DMatrix for GPU usage

# Predict the probability of interaction using the model
false_samples['Predicted_Prob'] = model_3.predict_proba(false_features)[:, 1]

# Sort the false samples based on the predicted probabilities
#false_samples_sorted = false_samples.sort_values(by='Predicted_Prob', ascending=False) # 98%-99%
false_samples_sorted = false_samples.sort_values(by='Predicted_Prob', ascending=True) # this selection is better 100%

# Sample an equal number of False samples as True samples
#false_samples_balanced = false_samples_sorted.sample(n=len(true_samples), random_state=42)
# Select the top `n` false samples based on the sorted predicted probabilities
false_samples_balanced = false_samples_sorted.head(len(true_samples))

# Drop the 'Predicted_Prob' column from the balanced false samples
false_samples_balanced = false_samples_balanced.drop(columns=['Predicted_Prob'])

# Combine balanced dataset
balanced_data = pd.concat([true_samples, false_samples_balanced])

# Debugging step: Check the shapes of the datasets
print(f"Shape of true_samples: {true_samples.shape}")
print(f"Shape of false_samples: {false_samples.shape}")
print(f"Shape of false_samples_sorted: {false_samples_sorted.shape}")
print(f"Shape of false_samples_balanced: {false_samples_balanced.shape}")
print(f"Shape of balanced_data: {balanced_data.shape}")

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

# Debugging step: Check the shape of final_data and some sample rows
print(f"Shape of final_data: {final_data.shape}")
print(f"Columns of final_data: {final_data.columns}")
print(f"First 5 rows of final_data:\n{final_data.head()}")

# Simplify the output file path
output_path = f"TF_{hm}_{version}_{model}_NV_2736_Type.csv"  # Simplified output path
print(output_path)
# Save the results to a new file
final_data.to_csv(output_path, index=False)

