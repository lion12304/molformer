import os
# use the sixth GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import pickle
import scanpy as sc
import anndata as ad
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from pertpy.data import srivatsan_2020_sciplex3

from helical import Geneformer,GeneformerConfig, UCE, UCEConfig, scGPT, scGPTConfig



CELL_TYPES = ['alveolar basal epithelial cells', 'lymphoblasts', 'mammary epithelial cells']
CELL_TYPE_DICT = {cell_type: i for i, cell_type in enumerate(CELL_TYPES)}

def get_highly_variable_gene(data_obj, n_top_genes=2000):
    # sc.pp.normalize_total(data_obj, target_sum=1e4)
    data_obj = data_obj.copy()
    sc.pp.log1p(data_obj)
    sc.pp.highly_variable_genes(data_obj, n_top_genes=n_top_genes)
    return data_obj.var.highly_variable

def prepare_sciplex_data(dose_value=10000, num_variable_genes=2000):
    data = srivatsan_2020_sciplex3()
    data_orig = data[data.obs['dose_value'] == dose_value].copy()
    data_orig.var.rename(columns={'ncounts': 'n_counts'}, inplace=True)
    data_orig.obs.rename(columns={'ncounts': 'n_counts'}, inplace=True)

    variable_genes = get_highly_variable_gene(data_orig, n_top_genes=num_variable_genes)
    # Available genes in the geneformer model
    with open("ensembl_mapping_dict_gc95M.pkl", "rb") as f:
        ensembl_mapping_dict = pickle.load(f)
        f.close()

    available_genes_keys = list(ensembl_mapping_dict.keys())
    variable_genes_list = data_orig.var.ensembl_id[variable_genes].values.tolist()
    available_variable_genes = [gene for gene in variable_genes_list if gene in set(available_genes_keys)]
    # create an index vector variable_genes to be True if ensembl_id is in available_variable_genes
    variable_genes = [True if gene in available_variable_genes else False for gene in data_orig.var.ensembl_id.values]
    variable_genes = np.array(variable_genes)

    # Filter data_orig with variable_genes
    data_orig = data_orig[:, variable_genes]
    print("Saving file - data_orig of shape: ", data_orig.shape)

    data_orig = data_orig[:, data_orig.var["ensembl_id"] != ""] 
    data_orig.obs["filter_pass"] = True
    data_orig.obs["cell_type"] = "unknown"

    # Save filtered data_orig
    print("Saving file - data_orig of shape: ", data_orig.shape)
    data_orig.write_h5ad(f"data_dose_{dose_value}_n_genes_{num_variable_genes}.h5ad")
    return data_orig

def setup_geneformer():
    geneformer_cfg = {
        "model_name": "gf-12L-30M-i2048",
        "batch_size": 256,
        "emb_layer": -1,
        "emb_mode": "cell",
        "device": "cuda:0",
        "accelerator": True
    }
    geneformer_config = GeneformerConfig(**geneformer_cfg)
    geneformer = Geneformer(configurer=geneformer_config)
    return geneformer

def setup_scgpt():
    scgpt_cfg = {
        "pad_token": "<pad>",
        "batch_size": 24,
        "fast_transformer": False,
        "nlayers": 12,
        "nheads": 8,
        "embsize": 512,
        "d_hid": 512,
        "dropout": 0.2,
        "n_layers_cls": 3,
        "mask_value": -1,
        "pad_value": -2,
        "world_size": 8,
        "accelerator": True,
        "device": "cuda:0",
        "use_fast_transformer": False
    }
    scgpt_config = scGPTConfig(**scgpt_cfg)
    scgpt = scGPT(configurer = scgpt_config)
    return scgpt

def setup_uce():
    model_config = UCEConfig(batch_size=10, device="cuda:0")
    uce = UCE(configurer=model_config)
    return uce

def get_model(model_name: str):
    if model_name == "geneformer":
        return setup_geneformer()
    elif model_name == "scgpt":
        return setup_scgpt()
    elif model_name == "uce":
        return setup_uce()
    else:
        raise ValueError("Invalid model name")

def process_data(ann_data, model_name):
    if model_name == "geneformer":
        dataset = model.process_data(ann_data, gene_names="ensembl_id")
    elif model_name == "scgpt":
        dataset = model.process_data(ann_data, gene_names="vocab")
    elif model_name == "uce":
        # Not working yet
        ann_data.var = ann_data.var.reindex(ann_data.var["vocab"])
        dataset = model.process_data(ann_data, gene_names="vocab")
    else:
        raise ValueError("Invalid model name")
    return dataset


model_name = "geneformer"
model = get_model(model_name)

ann_data = prepare_sciplex_data(dose_value=10., num_variable_genes=2000)

cellxgene_to_ensembl = pd.read_csv("gene_info.csv", index_col=0)
ensembl_to_vocab_mapping = cellxgene_to_ensembl[["feature_id", "feature_name"]].set_index("feature_id").to_dict()["feature_name"]
ann_data.var["vocab"] = ann_data.var.ensembl_id.map(ensembl_to_vocab_mapping)
print("Number of NaNs in vocab: ", ann_data.var["vocab"].isna().sum())

dataset = process_data(ann_data, model_name)

if os.path.exists(f"{model_name}_embeddings.npy"):
    embeddings = np.load("gene_embeddings.npy")
else:
    embeddings = model.get_embeddings(dataset)
    np.save("gene_embeddings.npy", embeddings)

cell_types = ann_data.obs['celltype'].apply(lambda x: CELL_TYPE_DICT[x])
nan_rows = np.isnan(embeddings).any(axis=1)
emb = embeddings[~nan_rows]
cell_types = cell_types[~nan_rows]
umap_emb = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine').fit_transform(emb)

plt.figure(figsize=(10, 10))
plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=cell_types, cmap='viridis', s=2)
plt.colorbar()
plt.title("UMAP embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
# plt.legend()
plt.savefig(f"umap_embeddings_emb_{model_name}_1.0.png")
