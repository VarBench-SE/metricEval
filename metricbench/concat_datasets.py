# dataset_test.push_to_hub("CharlyR/varbench", config_name=subset, split="test")
from datasets import (
    get_dataset_config_names,
    concatenate_datasets,
    Dataset,
    load_dataset
)
from huggingface_hub import login
import os
# 1rst program:
# pull dataset evaluation(all configs)
# merge it into one with a new id
# remove from original dataset all the data already pre
# add column
# push
# login(token=os.environ.get("HF_TOKEN"))

all_configs = get_dataset_config_names("CharlyR/varbench-evaluation")

all_datasets: list[Dataset] = []

for config in all_configs:
    all_datasets.append(load_dataset("CharlyR/varbench-evaluation", config, split="tikz"))


concat_datasets = concatenate_datasets(all_datasets)#works because all the metrics are the same, might eventually need to adapt it when more metrics are computed

print(concat_datasets)




""" dataset: Dataset = load_dataset(
    "CharlyR/varbench-evaluation", dataset_split, split="tikz"
)

filtered_df = filtered_df[ordered_metrics + ["id", "difficulty"]]
filtered_df

computed_metrics_names = [
    metric_name
    for metric_name in dataset.column_names
    if metric_name.endswith("Metric") and not metric_name.startswith("best")
] """


# 2nd programm
# pull dataset metric eval config untreated
# pull dataset metric-eval config treated
# remove from untreated all the treated ones
# launch app with a random one
