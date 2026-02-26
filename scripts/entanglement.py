import torch
import pickle
from itertools import product
from tqdm import tqdm

RESULTS_DIR = "results"
DATASETS = [
    "qasc",
    "wiki_qa",
    "quartz",
    "paws",
    # "story_cloze", # error
    "winogrande",
    "wsc",
]
MODEL_NAME = "t5-base"


def load_dct(dct_path):
    with open(dct_path, "rb") as f:
        return pickle.load(f)


def layer_name_fn(name):
    # if "attn" in name:
    #     return name.replace("attn", "attn.in_proj_weight")
    # else:
    #     return name + ".weight"
    return name + ".weight"


# Load a test sd to get the layer names
ci_dct = load_dct(f"{RESULTS_DIR}/covs_d{DATASETS[0]}_m{MODEL_NAME}.pkl")
layer_names = list(ci_dct["covs"].keys())

deltas = torch.zeros(len(layer_names), len(DATASETS), len(DATASETS), len(DATASETS))


T = len(DATASETS)
print(f"Computing entanglement for {MODEL_NAME} on {DATASETS}")
for t, i, j in tqdm(
    product(range(T), range(T), range(T)), total=T * T * T, desc="t,i,j"
):
    if j < i:
        continue

    mi_sd = load_dct(f"{RESULTS_DIR}/mats_d{DATASETS[i]}_m{MODEL_NAME}.pkl")
    mj_sd = load_dct(f"{RESULTS_DIR}/mats_d{DATASETS[j]}_m{MODEL_NAME}.pkl")
    ct_sd = load_dct(f"{RESULTS_DIR}/covs_d{DATASETS[t]}_m{MODEL_NAME}.pkl")["covs"]

    # Loop over layers
    for l in range(len(layer_names)):
        k, kp = layer_names[l], layer_name_fn(layer_names[l])
        mi = mi_sd[kp][1]
        mj = mj_sd[kp][1]
        ct = torch.from_numpy(ct_sd[k]).to(mi.dtype).to(mi.device)
        deltas[l, i, j, t] = torch.trace(mi @ ct @ mj.T)
        deltas[l, j, i, t] = torch.trace(mj @ ct @ mi.T)


# Save deltas
output_file = f"{RESULTS_DIR}/entanglements_m{MODEL_NAME}.pkl"
with open(output_file, "wb") as f:
    pickle.dump(deltas, f)
print(f"Entanglements saved to {output_file}")
