# # 1. Download the dataset
# python3 -c "from datasets import load_dataset; load_dataset('story_cloze', '2016')"

export HF_HOME=$SCRATCH/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# 2.Pre-download model and dataset
python scripts/download_model.py 


# 3. Train T5-Base on these
# "qasc": QASCReader,
# "wiki_qa": WikiQAReader,
# "quartz": QuaRTzReader,
# "paws": PAWSReader,
# "story_cloze": StoryClozeReader,
# "winogrande": WinograndeReader,
# "wsc": WSCReader,
python src/training.py -c configs/t5_base.json -k train_batch_size=8 gradient_accumulation_factor=1 project_name=training experiment_name=qasc train_dataset=qasc train_dataset_mixture=None inference_dataset_mixture=qasc


# 4. Evaluate T5-Base on these
dataset=quartz; \
path_to_checkpoint=$SCRATCH/ties/exp_out/training/t5-base/${dataset}/best_model.pt; \
eval_split=validation; \
python ./src/inference.py -c configs/t5_base.json -i ${dataset} --kwargs checkpoint_to_directly_load_model=${path_to_checkpoint} split=${eval_split} project_name=evaluation experiment_name=${dataset}


# 5. Merge T5-Base on these
datasets_to_merge=qasc,quartz; \
eval_split=validation; \
python ./src/ties_merging.py -c configs/t5_base.json -i ${datasets_to_merge} -m ${datasets_to_merge} -f basic_mean --kwargs split=${eval_split} project_name=evaluation experiment_name=mean