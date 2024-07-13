from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_dataset("LysandreJik/glue-mnli-train")
#dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)
#sliced_train_dataset = DatasetDict(dataset["train"][3952:3954])
#sliced_train_dataset = DatasetDict(dataset["train"][3949:3954])
#sliced_train_dataset = DatasetDict(dataset["train"][12403:12404])
sliced_train_dataset = pd.DataFrame(dataset["train"][:])
print(sliced_train_dataset.shape)
#sliced_train_dataset.reset_index(drop=True, inplace=True)

# Convert the dataset to a pandas DataFrame
#train_df = pd.DataFrame(train_dataset)

# Step 5: Split the dataset into train and validation sets
train_df, val_df = train_test_split(sliced_train_dataset , test_size=0.2, random_state=42)


print(train_df.head())
print('\n')
print(val_df.head())
print('\n\n')

# Step 6: Convert the DataFrames back to the Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(train_dataset)
print('\n')
print(val_dataset)
print('\n\n')
# Print the resulting datasets
#print(train_dataset['label'])
#print(train_dataset['premise'])
#print(train_dataset['hypothesis'])
#print(val_dataset)
#print(sliced_train_dataset['premise'])
#print(sliced_train_dataset['premise','hypothesis'])