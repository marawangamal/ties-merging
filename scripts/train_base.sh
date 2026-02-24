# 3. Train T5-Base on these
# "qasc": QASCReader,
# "wiki_qa": WikiQAReader,
# "quartz": QuaRTzReader,
# "paws": PAWSReader,
# "story_cloze": StoryClozeReader,
# "winogrande": WinograndeReader,
# "wsc": WSCReader,
model_name=t5_large
for dataset in qasc wiki_qa quartz paws story_cloze winogrande wsc; do
  echo "Training on ${dataset}..."
  python src/training.py -c configs/${model_name}.json -k project_name=training experiment_name=${dataset} train_dataset=${dataset} train_dataset_mixture=None inference_dataset_mixture=${dataset}
done
