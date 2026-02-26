import sys
import os
import re
from tqdm import tqdm

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import os
import torch
import numpy as np


# ---------------------------------------------------------------------------
#  Basic Merging Methods
# ---------------------------------------------------------------------------


def merge_ta(pt_tensor, task_tensors, scaling_coefficient=1.0, **kwargs):
    return pt_tensor + task_tensors.sum(dim=0) * scaling_coefficient


def merge_wa(pt_tensor, task_tensors, **kwargs):
    return pt_tensor + task_tensors.mean(dim=0)


def merge_mix_alpha(
    pt_tensor,
    task_tensors,
    key,
    pattern="",
    alpha_pattern=1.0,
    scaling_coefficient=1.0,
    **kwargs,
):
    alpha = scaling_coefficient
    if pattern and re.search(pattern, key):
        alpha = alpha_pattern
    print(f"[{key}] Using alpha={alpha}")
    return pt_tensor + (alpha * task_tensors.sum(dim=0))


# ---------------------------------------------------------------------------
#  TSV
# ---------------------------------------------------------------------------


def _compute_procrustes(x: torch.Tensor) -> torch.Tensor:
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    return u @ vt


def merge_tsv(pt_tensor, task_tensors, **kwargs):
    """Computes the TSV merge of the given tensors.

    Computes: Uo  Dc Vto

    Args:
        pt_tensor (torch.Tensor): The pretrained tensor. Shape: (Di, Do)
        task_tensors (torch.Tensor): The tasktensors to merge. Shape: (N_tasks, Di, Do)

    Returns:
        torch.Tensor: The merged tensors. Shape: (Di, Do)
    """
    N_tasks = len(task_tensors)
    u, s, vt = torch.linalg.svd(task_tensors, full_matrices=False)
    R = min(u.shape[1], vt.shape[2])
    Rp = R // N_tasks
    u, s, vt = u[:, :, :Rp], s[:, :Rp], vt[:, :Rp, :]

    # # # w/o decorrelation
    # tau_bl = torch.einsum("bij,bj,bjk->bik", u, s, vt)
    # tau[layer_name] = tau_bl.sum(dim=0)

    # w/ decorrelation
    B, Di, _ = u.shape
    _, _, Do = vt.shape
    # (Di, B, R)
    u_hat = u.permute(1, 0, 2).reshape(Di, B * Rp)
    s_hat = s.reshape(-1)
    vt_hat = vt.reshape(B * Rp, Do)
    u_ortho = _compute_procrustes(u_hat)  # (Di, Rp)
    vt_ortho = _compute_procrustes(vt_hat.T).T  # (Rp, Do)
    tau_l = torch.einsum("ij,j,jk->ik", u_ortho, s_hat, vt_ortho)  # (Di, Do)
    return pt_tensor + tau_l


# ---------------------------------------------------------------------------
#  ISOC
# ---------------------------------------------------------------------------


def merge_isoc(pt_tensor, task_tensors, mode="spectral", *args, **kwargs):
    # m = task_tensors.sum(dim=0)
    # u, s, vt = torch.linalg.svd(m, full_matrices=False)
    # s_mean = s.mean() * torch.ones_like(s)
    # return pt_tensor + torch.einsum("ik,k,kj->ij", u, s_mean, vt)
    m = task_tensors.sum(dim=0)
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    if mode == "mean":
        s_iso = s.mean() * torch.ones_like(s)
    elif mode == "unity":
        s_iso = torch.ones_like(s)
    elif mode == "rms":
        s_iso = torch.sqrt((s**2).mean()) * torch.ones_like(s)
    elif mode == "spectral":
        s_iso = s[0] * torch.ones_like(s)  # Use largest singular value
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return pt_tensor + torch.einsum("ik,k,kj->ij", u, s_iso, vt)


# ---------------------------------------------------------------------------
#  RegMean
# ---------------------------------------------------------------------------
# Avg.,qasc,wiki_qa,quartz,paws,story_cloze,winogrande,wsc
# 73.3, 97.8, 95.4, 75.5, 84.6, 51.2, 57.5, 51.4


def _param_key_to_module_key(key: str):
    # Example:
    # param: transformer.encoder.block.11.layer.1.DenseReluDense.wo.weight
    # module: transformer.encoder.block.0.layer.0.SelfAttention.q
    return key.replace(".weight", "")


def merge_regmean(pt_tensor, task_tensors, key, covariance_paths=None, **kwargs):

    if covariance_paths is None or len(covariance_paths) == 0:
        raise ValueError("covariance_paths is required for regmean merging")

    c = []
    for cov_path in covariance_paths:
        with np.load(cov_path) as cdict:
            kp = _param_key_to_module_key(key)
            if kp not in cdict:
                print(f"[skipped] {kp} not found in {cov_path}")
                return task_tensors.mean(dim=0) + pt_tensor
            c.append(cdict[kp])
    c = torch.stack(
        [
            torch.as_tensor(x, device=task_tensors.device, dtype=task_tensors.dtype)
            for x in c
        ]
    )
    print("[RegMean] Successfully loaded covariances for {key}")
    return ((task_tensors @ c).sum(dim=0) @ torch.linalg.pinv(c.sum(dim=0))) + pt_tensor


def opmerge(
    pt_state_dict,
    ft_ckpt_paths,
    merge_type,
    remove_keys,
    cache_size=32,
    scaling_coefficient=1.0,
    covariance_paths=None,
    **merge_kwargs,
):

    merge_fn = {
        "wa": merge_wa,
        "ta": merge_ta,
        "tsv": merge_tsv,
        "isoc": merge_isoc,
        "mix_alpha": merge_mix_alpha,
        "regmean": merge_regmean,
    }[merge_type]

    new_sd = {}
    # Swap: pretrained
    # pt_state_dict = mhas.copy_from_pytorch_state_dict(pt_state_dict)
    num_keys = len(pt_state_dict)
    cache = {}
    keys = list(pt_state_dict.keys())

    for i, (key, pt_tens) in enumerate(pt_state_dict.items()):
        if len(pt_tens.shape) == 2 and not key in remove_keys:
            # merge
            # load the layer from each of the ft_ckpt_paths
            task_tensors = []
            for ft_ckpt_path in ft_ckpt_paths:
                if ft_ckpt_path in cache and key in cache[ft_ckpt_path]:
                    # cache hit
                    ft_state_dict = cache[ft_ckpt_path]
                else:
                    # cache miss
                    ft_state_dict = torch.load(ft_ckpt_path)
                    # Swap: finetuned
                    # ft_state_dict = mhas.copy_from_pytorch_state_dict(ft_state_dict)
                    cache[ft_ckpt_path] = {}  # empty cache
                    for k in keys[i + 1 : i + cache_size + 1]:  # cache items
                        cache[ft_ckpt_path][k] = ft_state_dict[k]
                task_tensors.append(ft_state_dict[key] - pt_tens)
            task_tensors = torch.stack(task_tensors)
            new_sd[key] = merge_fn(
                pt_tensor=pt_tens,
                task_tensors=task_tensors,
                key=key,
                scaling_coefficient=scaling_coefficient,
                covariance_paths=covariance_paths,
                **merge_kwargs,
            )
            print(
                f"[{i}/{num_keys}] Merged {key} with {merge_type} (lambda={scaling_coefficient})"
            )
        else:
            # keep as is
            new_sd[key] = pt_tens
            print(f"[{i}/{num_keys}] Keeping {key} as is")

    # Unswap: pretrained
    # new_sd = mhas.copy_to_pytorch_state_dict(new_sd)
    return new_sd
