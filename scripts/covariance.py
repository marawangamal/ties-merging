import argparse
import os
import os.path as osp
import pickle
import sys
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# set environment variables
os.environ["HF_HOME"] = os.path.join(os.environ.get("SCRATCH", ""), "huggingface")

from src.data.dataset_readers import DATASET_CLASSES
from src.data.Batcher import Batcher
from src.data.PytorchDataset import PytorchDataset
from src.model.T5Wrapper import T5Wrapper


# Running covariance computation


class OnlineCovariance:
    """
    A class to calculate the mean and the covariance matrix
    of the incrementally added, n-dimensional data.
    """

    def __init__(self, order):
        """
        Parameters
        ----------
        order: int, The order (=="number of features") of the incrementally added
        dataset and of the resulting covariance matrix.
        """
        self._order = order
        self._shape = (order, order)
        self._identity = np.identity(order)
        self._ones = np.ones(order)
        self._count = 0
        self._mean = np.zeros(order)
        self._cov = np.zeros(self._shape)

    @property
    def count(self):
        """
        int, The number of observations that has been added
        to this instance of OnlineCovariance.
        """
        return self._count

    @property
    def mean(self):
        """
        double, The mean of the added data.
        """
        return self._mean

    @property
    def cov(self):
        """
        array_like, The covariance matrix of the added data.
        """
        return self._cov

    @property
    def corrcoef(self):
        """
        array_like, The normalized covariance matrix of the added data.
        Consists of the Pearson Correlation Coefficients of the data's features.
        """
        if self._count < 1:
            return None
        variances = np.diagonal(self._cov)
        denomiator = np.sqrt(variances[np.newaxis, :] * variances[:, np.newaxis])
        return self._cov / denomiator

    def add(self, observation):
        """
        Add the given observation to this object.

        Parameters
        ----------
        observation: array_like, The observation to add.
        """
        if self._order != len(observation):
            raise ValueError(f"Observation to add must be of size {self._order}")

        self._count += 1
        delta_at_nMin1 = np.array(observation - self._mean)
        self._mean += delta_at_nMin1 / self._count
        weighted_delta_at_n = np.array(observation - self._mean) / self._count
        shp = (self._order, self._order)
        D_at_n = np.broadcast_to(weighted_delta_at_n, self._shape).T
        D = (delta_at_nMin1 * self._identity).dot(D_at_n.T)
        self._cov = self._cov * (self._count - 1) / self._count + D

    def merge(self, other):
        """
        Merges the current object and the given other object into a new OnlineCovariance object.

        Parameters
        ----------
        other: OnlineCovariance, The other OnlineCovariance to merge this object with.

        Returns
        -------
        OnlineCovariance
        """
        if other._order != self._order:
            raise ValueError(
                f"""
                   Cannot merge two OnlineCovariances with different orders.
                   ({self._order} != {other._order})
                   """
            )

        merged_cov = OnlineCovariance(self._order)
        merged_cov._count = self.count + other.count
        count_corr = (other.count * self.count) / merged_cov._count
        merged_cov._mean = (
            self.mean / other.count + other.mean / self.count
        ) * count_corr
        flat_mean_diff = self._mean - other._mean
        shp = (self._order, self._order)
        mean_diffs = np.broadcast_to(flat_mean_diff, self._shape).T
        merged_cov._cov = (
            self._cov * self.count
            + other._cov * other._count
            + mean_diffs * mean_diffs.T * count_corr
        ) / merged_cov.count
        return merged_cov


def main(args):
    # HPs
    DATASET_NAME = args.dataset
    MODEL_NAME = args.model
    MAX_HOOKS = args.max_hooks
    MAX_BATCHES = args.max_batches
    RESULTS_DIR = "results"
    covs_filepath = os.path.join(RESULTS_DIR, f"covs_d{DATASET_NAME}_m{MODEL_NAME}.pkl")
    spectrums_filepath = os.path.join(
        RESULTS_DIR, f"spectrums_d{DATASET_NAME}_m{MODEL_NAME}.pkl"
    )

    root_dir = osp.join(
        os.environ.get("SCRATCH", ""), "ties", "exp_out", "training", MODEL_NAME
    )
    task_to_model_dict = {
        "qasc": osp.join(root_dir, "qasc", "best_model.pt"),
        "quartz": osp.join(root_dir, "quartz", "best_model.pt"),
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, device)

    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    dataset_reader = DATASET_CLASSES[DATASET_NAME](dataset_kwargs)
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=32,
        eval_batchSize=32,
        world_size=1,
        device=0,
    )
    train_iterator = batcher.get_trainBatches("train", 0)

    # Init model
    transformer = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = T5Wrapper(transformer, tokenizer)

    # Optionally load state_dict
    model.load_state_dict(torch.load(task_to_model_dict[DATASET_NAME]))
    model.to(device)
    model.train()

    # Compute running covariance of activations
    stats = {}  # layer_name -> hook_result
    handles = []  # references to hooks

    def hook(name):
        # Hook gets (module, input, output)
        def h(mod, _inp, out):
            # print("out.shape", out.shape)
            inp = _inp[0]
            if torch.is_tensor(inp):
                print("inp.shape", inp.shape)
                B, T, D = inp.shape
                if name not in stats:
                    ocov = OnlineCovariance(D)
                    stats[name] = ocov
                ocov = stats[name]
                for i in range(B):
                    j = torch.randint(0, T, (1,)).item()
                    v = inp[i, j].cpu().detach().numpy()
                    ocov.add(v)
                stats[name] = ocov

        return h

    num_registered = 0
    for i, (module_name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Linear):
            print("Registering hook for", module_name)
            h = m.register_forward_hook(hook(module_name))
            handles.append(h)
            if num_registered > MAX_HOOKS:
                break
            num_registered += 1

    print("Registered", num_registered, "hooks")

    # Run forward passes
    model.to(device)
    train_iterator = batcher.get_trainBatches("train", 0)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_iterator), total=MAX_BATCHES):
            model(batch)
            if i > MAX_BATCHES:
                break

    # Clear hooks
    for h in handles:
        h.remove()

    # Save covs and means

    os.makedirs(RESULTS_DIR, exist_ok=True)

    covs = {}
    means = {}
    spectrums = {}
    for s in stats:
        c = stats[s].cov
        m = stats[s].mean
        print(s, c.shape)
        covs[s] = c
        means[s] = m
        spectrums[s] = np.linalg.svdvals(c)

    with open(covs_filepath, "wb") as f:
        pickle.dump(
            {
                "covs": covs,
                "means": means,
            },
            f,
        )

    # Save cov spectrums separately
    with open(spectrums_filepath, "wb") as f:
        pickle.dump(spectrums, f)


if __name__ == "__main__":
    DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="qasc")
    parser.add_argument("--model", type=str, default="t5-base")
    parser.add_argument("--max-hooks", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset == "ALL":
        for dataset in DATASETS:
            args.dataset = dataset
            print(f"Computing covariances and spectrums for {dataset} and {args.model}")
            main(args)
            print(
                f"Done computing covariances and spectrums for {dataset} and {args.model}"
            )
    else:
        main(args)
