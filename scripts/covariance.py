import argparse
import os
import sys
from typing import Dict
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["HF_HOME"] = os.path.join(os.environ.get("SCRATCH", ""), "huggingface")

from src.data.dataset_readers import DATASET_CLASSES
from src.data.Batcher import Batcher
from src.data.PytorchDataset import PytorchDataset
from src.model.T5Wrapper import T5Wrapper
from src.covariance import OnlineCovariance, register_hooks


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    dataset_reader = DATASET_CLASSES[args.dataset](dataset_kwargs)
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=args.batch_size,
        eval_batchSize=args.batch_size,
        world_size=1,
        device=0,
    )
    train_iterator = batcher.get_trainBatches("train", 0)

    # Init model
    transformer = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = T5Wrapper(transformer, tokenizer)
    model.load_state_dict(torch.load(args.ft_ckpt_path))
    model.to(device)
    model.eval()

    mask_ref = [None, None]
    cobjs, handles = register_hooks(
        model, cov_device=args.cov_device, mask_ref=mask_ref
    )

    # Run forward passes
    train_iterator = batcher.get_trainBatches("train", 0)
    print(f"Running forward passes for {args.num_batches} batches")
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(train_iterator), total=args.num_batches, leave=False
        ):
            mask_ref[0] = batch["input_mask"]
            mask_ref[1] = batch["target_mask"]
            model(batch)
            if i > args.num_batches:
                break

    # Clear hooks
    for h in handles:
        h.remove()

    return cobjs


if __name__ == "__main__":
    DATASETS = [
        "qasc",
        "wiki_qa",
        "quartz",
        "paws",
        "story_cloze",
        "winogrande",
        "wsc",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qasc")
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ss_num_batches = args.num_batches
    ss_batch_size = args.batch_size
    cov_dir = f"results/{args.model}/covariances_n{ss_num_batches}_b{ss_batch_size}"
    os.makedirs(cov_dir, exist_ok=True)

    for dataset in DATASETS:
        # HACK: set device for model and covariance
        args.dataset = dataset
        args.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.cov_device = torch.device("cpu")
        cov_filepath = os.path.join(cov_dir, f"covariance_{dataset}.npz")

        # pt_state_dict = model.state_dict()
        args.ft_ckpt_path = os.path.join(
            "exp_out", "training", args.model, dataset, "best_model.pt"
        )
        remove_keys = [
            "transformer.encoder.embed_tokens.weight",
            "transformer.decoder.embed_tokens.weight",
        ]

        # First check if the covariances and spectrums have already been computed
        if os.path.exists(cov_filepath) and not args.overwrite:
            print(f"Covariances already computed for {args.dataset} and {args.model}")
            continue

        print(f"Computing covariances and spectrums for {dataset} and {args.model}")
        cobjs = main(args)

        # Convert cobjs to saveable arrays (np.savez can't save tuples of inhomogeneous shapes)
        # two loops to save memory
        for lname in list(cobjs):
            cobjs[f"{lname}_n"] = cobjs[lname].n
        cobjs: Dict = {
            k: v.cov.cpu().numpy() if isinstance(v, OnlineCovariance) else v
            for k, v in cobjs.items()
        }

        # Save covariance using np.savez
        np.savez(cov_filepath, **cobjs)
        print(f"Covariance saved to {cov_filepath}")
