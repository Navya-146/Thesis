import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from rdkit.Chem import MolFromSmarts
from tokenizers import Tokenizer

from main_code.FGR.load_FGR import get_representation

PROJECT_PATH = "/home/workspace/da24c011"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------
# Dataset
# --------------------------
class DrugOmicsIC50Dataset(Dataset):
    def __init__(self, df, omics_data, drug_dict, tokenizer, fgroups_list,
                 omics_input_sizes, task="regression", ic50_threshold=None):
        self.df = df.reset_index(drop=True)
        self.omics = omics_data
        self.drugs = {str(k).strip().lower(): v for k, v in drug_dict.items()}
        self.tokenizer = tokenizer
        self.fgroups_list = fgroups_list
        self.omics_input_sizes = omics_input_sizes
        self.task = task
        self.threshold = ic50_threshold

        # -------------------
        # Precompute drug embeddings on CPU
        # -------------------
        unique_drugs = list(set([str(d).strip().lower() for d in df["DRUG_NAME"]]))
        self.drug_embeddings = {}
        for smiles in unique_drugs:
            rep = get_representation([self.drugs[smiles]], "FGR", self.fgroups_list, self.tokenizer)
            self.drug_embeddings[smiles] = torch.from_numpy(rep).float()  # CPU tensor

        # Precompute targets and omics indices for fast __getitem__
        self.targets = []
        self.row_omics = []
        for _, row in self.df.iterrows():
            cell_id = row["Demap id"]
            # Omics vectors as CPU tensors
            omics_views = []
            for modality in ["exp", "cnv", "mut", "meth"]:
                vec = list(self.omics[modality][cell_id].values())
                omics_views.append(torch.tensor(vec, dtype=torch.float32))
            self.row_omics.append(omics_views)

            # Target
            if self.task == "regression":
                t = torch.tensor(row["LN_IC50"], dtype=torch.float32)
            else:
                t = torch.tensor(int(row["LN_IC50"] <= self.threshold), dtype=torch.long)
            self.targets.append(t)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drug_name = str(row["DRUG_NAME"]).strip().lower()
        drug_rep = self.drug_embeddings[drug_name]  # CPU tensor
        omics_views = self.row_omics[idx]            # CPU tensors
        target = self.targets[idx]
        return (drug_rep, *omics_views, target)


# --------------------------
# DataModule
# --------------------------
class DrugOmicsDataModule(LightningDataModule):
    def __init__(self, data_dir, train_val_test_split, batch_size,
                 num_workers, omics_input_sizes,
                 task="regression", ic50_threshold=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.threshold = ic50_threshold
        self.omics_input_sizes = omics_input_sizes

        # --- Load omics data ---
        self.omics = {
            "exp": pd.read_csv(os.path.join(PROJECT_PATH, "raw data from colab", "expression_wo_cosmic.csv"), index_col=0).to_dict(orient="index"),
            "mut": pd.read_csv(os.path.join(PROJECT_PATH, "raw data from colab", "mutation_wo_cosmic.csv"), index_col=0).to_dict(orient="index"),
            "meth": pd.read_csv(os.path.join(PROJECT_PATH, "raw data from colab", "methylation_wo_mean_cosmic.csv"), index_col=0).to_dict(orient="index"),
            "cnv": pd.read_csv(os.path.join(PROJECT_PATH, "raw data from colab", "cnv_wo_cosmic.csv"), index_col=0).to_dict(orient="index"),
        }

        # --- Drug SMILES dictionary ---
        drug_smiles_df = pd.read_csv(
            os.path.join(PROJECT_PATH, "data to be used", "drug_smiles.csv"),
            sep="\t", index_col=0
        )
        self.drug_dict = drug_smiles_df["SMILES"].to_dict()

        # --- Functional groups + tokenizer ---
        fgroups = pd.read_parquet(os.path.join(BASE_DIR, "fg.parquet"))["SMARTS"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in fgroups]
        self.tokenizer = Tokenizer.from_file(
            os.path.join(BASE_DIR, "tokenizers", "BPE_pubchem_500.json")
        )

        # --- IC50 dataset ---
        self.full_df = pd.read_csv(os.path.join(PROJECT_PATH, "data to be used", "fin_ic50_21-7.csv"))

        # Split params
        self.split = train_val_test_split
        self.train_set, self.valid_set, self.test_set = None, None, None

    def setup(self, stage=None):
        if self.train_set is None:
            df = self.full_df.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(df)
            train_end = int(n * self.split["train"])
            val_end = train_end + int(n * self.split["val"])

            train_df = df.iloc[:train_end].reset_index(drop=True)
            val_df = df.iloc[train_end:val_end].reset_index(drop=True)
            test_df = df.iloc[val_end:].reset_index(drop=True)

            self.train_set = DrugOmicsIC50Dataset(
                train_df, self.omics, self.drug_dict,
                self.tokenizer, self.fgroups_list,
                self.omics_input_sizes, self.task, self.threshold
            )
            self.valid_set = DrugOmicsIC50Dataset(
                val_df, self.omics, self.drug_dict,
                self.tokenizer, self.fgroups_list,
                self.omics_input_sizes, self.task, self.threshold
            )
            self.test_set = DrugOmicsIC50Dataset(
                test_df, self.omics, self.drug_dict,
                self.tokenizer, self.fgroups_list,
                self.omics_input_sizes, self.task, self.threshold
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=False
        )
