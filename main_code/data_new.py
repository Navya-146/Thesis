import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import pandas as pd
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer
from omegaconf import OmegaConf
import os

PROJECT_PATH = "/home/da24c011/miniconda3/project/"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script lives
# Your imports for FGR model and get_representation
from main_code.FGR.load_FGR import get_fgr_model, get_representation

# -----------------------
# Dataset
# -----------------------
class DrugOmicsIC50Dataset(Dataset):
    def __init__(self, df, drug_encoder, omics_data, drug_dict, tokenizer, 
                 fgroups_list, gene_orders, omics_input_sizes):
        self.df = df.reset_index(drop=True)
        self.encoder = drug_encoder.eval()
        self.omics = omics_data  # dict of modality_name -> dataframe or array
        self.drugs = {str(k).strip().lower(): v for k, v in drug_dict.items()}
        self.tokenizer = tokenizer
        self.fgroups_list = fgroups_list
        self.gene_orders = gene_orders
        self.omics_input_sizes = omics_input_sizes  # [exp_dim, cnv_dim, mut_dim, meth_dim]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- Drug encoding ----
        drug_name = str(row["DRUG_NAME"]).strip().lower()
        if drug_name not in self.drugs:
            raise KeyError(f"Drug '{drug_name}' not found in drug_dict.")
        smiles = self.drugs[drug_name]

        drug_rep = get_representation([smiles], "FGR", self.fgroups_list, self.tokenizer)

        # ---- Omics encoding (separate branches) ----
        omics_views = []
        for size, modality in zip(self.omics_input_sizes, ["exp", "cnv", "mut", "meth"]):
            if modality not in self.omics:
                raise KeyError(f"Omics modality '{modality}' missing from omics_data.")
            omics_dict = self.omics[modality][row["Demap id"]]
            omics_vector = torch.tensor(list(omics_dict.values()), dtype=torch.float32)
            omics_views.append(omics_vector)

        # ---- Target ----
        target = torch.tensor(row["LN_IC50"], dtype=torch.float32)

        return torch.from_numpy(drug_rep).float(), *omics_views, target





# -----------------------
# DataModule
# -----------------------
class DrugOmicsDataModule(LightningDataModule):
    def __init__(self, data_dir, drug_encoder, train_val_test_split, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.drug_encoder = drug_encoder.eval()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load omics data once here
        self.omics = {}
        self.gene_orders = {}

        # Load expression
        df_expr = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "expression_wo_cosmic.csv"))), index_col=0)
        self.omics["expr"] = df_expr.to_dict(orient="index")
        self.gene_orders["expr"] = list(df_expr.columns)

        # Load mutation
        df_mut = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "mutation_wo_cosmic.csv"))), index_col=0)
        self.omics["mut"] = df_mut.to_dict(orient="index")
        self.gene_orders["mut"] = list(df_mut.columns)

        # Load methylation
        df_meth = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "methylation_wo_mean_cosmic.csv"))), index_col=0)
        self.omics["meth"] = df_meth.to_dict(orient="index")
        self.gene_orders["meth"] = list(df_meth.columns)

        # Load CNV
        df_cnv = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "cnv_wo_cosmic.csv"))), index_col=0)
        self.omics["cnv"] = df_cnv.to_dict(orient="index")
        self.gene_orders["cnv"] = list(df_cnv.columns)




        # Load drug SMILES dictionary
        drug_smiles_df = pd.read_csv((os.path.join(PROJECT_PATH, "data to be used", "drug_smiles.csv")), sep="\t", index_col=0)
        self.drug_dict = drug_smiles_df["SMILES"].to_dict()

        # Load functional groups and tokenizer
        fgroups = pd.read_parquet((os.path.join(BASE_DIR, "fg.parquet")))["SMARTS"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in fgroups]
        self.tokenizer = Tokenizer.from_file((os.path.join(BASE_DIR,"tokenizers", "BPE_pubchem_500.json")))

        # Load full dataframe with drug, cell line, IC50
        self.full_df = pd.read_csv((os.path.join(PROJECT_PATH, "data to be used", "fin_ic50_21-7.csv")))

        # Train/val/test split proportions
        self.split = train_val_test_split

        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def setup(self, stage=None):
        if not self.train_set and not self.valid_set and not self.test_set:
            df = self.full_df.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(df)
            train_end = int(n * self.split["train"])
            val_end = train_end + int(n * self.split["val"])

            train_df = df.iloc[:train_end].reset_index(drop=True)
            val_df = df.iloc[train_end:val_end].reset_index(drop=True)
            test_df = df.iloc[val_end:].reset_index(drop=True)

            self.train_set = DrugOmicsIC50Dataset(
                train_df, self.drug_encoder, self.omics, self.drug_dict, self.tokenizer, self.fgroups_list, self.gene_orders
            )
            self.valid_set = DrugOmicsIC50Dataset(
                val_df, self.drug_encoder, self.omics, self.drug_dict, self.tokenizer, self.fgroups_list, self.gene_orders
            )
            self.test_set = DrugOmicsIC50Dataset(
                test_df, self.drug_encoder, self.omics, self.drug_dict, self.tokenizer, self.fgroups_list, self.gene_orders
            )


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# -----------------------
# Usage example
# -----------------------
if __name__ == "__main__":
    data_dir = "final_ish/data to be used"
    batch_size = 32
    num_workers = 0
    split = {"train": 0.7, "val": 0.15, "test": 0.15}

    # Load pretrained FGR encoder
    encoder = get_fgr_model("final_ish/checkpoints/epoch_000_val_0.8505.ckpt")

    # Instantiate DataModule
    data_module = DrugOmicsDataModule(data_dir, encoder, split, batch_size, num_workers)

    # Prepare datasets
    data_module.setup()

    # Access loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Iterate one batch to test
    for batch in train_loader:
        drug_vec, omics_views, label = batch
        print("Drug vector shape:", drug_vec.shape)
        print("Number of omics views:", len(omics_views))
        print("Label shape:", label.shape)
        break
