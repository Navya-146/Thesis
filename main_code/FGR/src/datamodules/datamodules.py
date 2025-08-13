import os
from typing import Dict, List, Optional, Union

import pandas as pd
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer

from final_ish.dataset import DrugOmicsIC50Dataset  # import your dataset class
from final_ish.load_fgr import get_representation  # if needed for initialization

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

class DrugOmicsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_encoder: torch.nn.Module,
        train_val_test_split: Dict[str, float],  # e.g. {'train': 0.7, 'val': 0.15, 'test': 0.15}
        batch_size: int,
        num_workers: int,
        loaders: DictConfig,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.drug_encoder = drug_encoder.eval()  # use pretrained encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cfg_loaders = loaders

        # Load your omics data here or pass as arguments if preferred
        expression_df = pd.read_csv("final_ish/raw data from colab/expression_wo_cosmic.csv", index_col=0)
        mutation_df = pd.read_csv("final_ish/raw data from colab/mutation_wo_cosmic.csv", index_col=0)
        methylation_df = pd.read_csv("final_ish/raw data from colab/methylation_wo_cosmic.csv", index_col=0)
        cnv_df = pd.read_csv("final_ish/raw data from colab/cnv_wo_cosmic.csv", index_col=0)    

        self.omics = {
            "expr": expression_df.to_dict(orient="index"),
            "mut": mutation_df.to_dict(orient="index"),
            "meth": methylation_df.to_dict(orient="index"),
            "cnv": cnv_df.to_dict(orient="index"),
        }

        drug_smiles_df = pd.read_csv("final_ish/data to be used/drug_smiles.tsv", sep="\t", index_col= 0)
        self.drug_dict = drug_smiles_df['SMILES'].to_dict()

        fgroups = pd.read_parquet("final_ish/data to be used/fg.parquet")["SMARTS"].tolist()
        self.fgroups_list = [MolFromSmarts(x) for x in fgroups]

        self.tokenizer = Tokenizer.from_file("final_ish/tokenizers/BPE_pubchem_500.json")

        # Load your main dataframe (the combined data with drug, cell line, IC50)
        self.full_df = pd.read_csv("final_ish/data to be used/fin_ic50_21-7.csv")

        # Split proportions
        self.split = train_val_test_split

        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # Split the dataframe into train/val/test once
        if not self.train_set and not self.valid_set and not self.test_set:
            train_frac = self.split.get("train", 0.7)
            val_frac = self.split.get("val", 0.15)
            test_frac = self.split.get("test", 0.15)

            # Shuffle and split your dataframe
            df = self.full_df.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(df)
            train_end = int(n * train_frac)
            val_end = train_end + int(n * val_frac)

            train_df = df.iloc[:train_end].reset_index(drop=True)
            val_df = df.iloc[train_end:val_end].reset_index(drop=True)
            test_df = df.iloc[val_end:].reset_index(drop=True)

            self.train_set = DrugOmicsIC50Dataset(
                train_df, self.drug_encoder, self.omics, self.drug_dict, self.tokenizer, self.fgroups_list
            )
            self.valid_set = DrugOmicsIC50Dataset(
                val_df, self.drug_encoder, self.omics, self.drug_dict, self.tokenizer, self.fgroups_list
            )
            self.test_set = DrugOmicsIC50Dataset(
                test_df, self.drug_encoder, self.omics, self.drug_dict, self.tokenizer, self.fgroups_list
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
