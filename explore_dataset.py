from datasets import load_dataset
from datasets import DatasetDict


dataset = load_dataset("LysandreJik/glue-mnli-train")
dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)
#sliced_train_dataset = DatasetDict(dataset["train"][3952:3954])
#sliced_train_dataset = DatasetDict(dataset["train"][3949:3954])
sliced_train_dataset = DatasetDict(dataset["train"][12403:12404])

print(sliced_train_dataset)