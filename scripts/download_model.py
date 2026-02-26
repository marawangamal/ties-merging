import os
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
from evaluate import load

# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# huggingFace_data = load_dataset(
#     "super_glue",
#     "rte",
#     split="train",
# )
metric = load("accuracy")

# DEBUG: dataset_stash ('super_glue', 'rte') | load_split train
