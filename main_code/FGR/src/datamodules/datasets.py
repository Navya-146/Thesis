import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import pandas as pd
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer


from main_code.FGR.load_FGR import get_representation
fgroups = pd.read_parquet("final_ish/data to be used/fg.parquet")[
    "SMARTS"
].tolist()
fgroups_list = [MolFromSmarts(x) for x in fgroups]
tokenizer = Tokenizer.from_file("final_ish/tokenizers/BPE_pubchem_500.json")
print("Tokenizer vocab size:", tokenizer.get_vocab_size())
print("Number of functional groups:", len(fgroups_list))

input_dim = tokenizer.get_vocab_size() + len(fgroups_list)
print("Total input dim for encoder:", input_dim)


expression_df = pd.read_csv("final_ish/raw data from colab/expression_wo_cosmic.csv", index_col=0)
mutation_df = pd.read_csv("final_ish/raw data from colab/mutation_wo_cosmic.csv", index_col=0)
methylation_df = pd.read_csv("final_ish/raw data from colab/methylation_wo_cosmic.csv", index_col=0)
cnv_df = pd.read_csv("final_ish/raw data from colab/cnv_wo_cosmic.csv", index_col=0)

expression = expression_df.to_dict(orient="index")
mutation = mutation_df.to_dict(orient="index")
methylation = methylation_df.to_dict(orient="index")
cnv = cnv_df.to_dict(orient="index")


drug_smiles_df = pd.read_csv("final_ish/data to be used/drug_smiles.tsv", sep="\t", index_col= 0)
drug_dict = drug_smiles_df['SMILES'].to_dict()
gdsc= pd.read_csv("final_ish/data to be used/fin_ic50_21-7.csv")


class DrugOmicsIC50Dataset(Dataset):
    def __init__(self, df, drug_encoder, omics_data, drug_dict, tokenizer, fgroups_list):
        """
        df: dataframe with columns ['Demap id', 'DRUG_NAME', 'LN_IC50']
        drug_encoder: pretrained FGR encoder
        omics_data: dict with keys: 'expr', 'mut', 'meth', 'cnv' → each maps cell_line_id to feature vector
        drug_dict: smiles and drugs
        tokenizer, fgroups_list: used in get_representation
        """
        self.df = df.reset_index(drop=True)
        self.encoder = drug_encoder
        self.omics = omics_data
        self.drugs = {str(k).strip().lower(): v for k, v in drug_dict.items()}
        self.tokenizer = tokenizer
        self.fgroups_list = fgroups_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        cell_line = row["Demap id"]
        drug_name = str(row["DRUG_NAME"]).strip().lower()

        if drug_name not in self.drugs:
            raise KeyError(
            f"Drug '{drug_name}' not found in drug_dict.\n"
            f"Sample keys: {list(self.drugs.keys())[:5]}"
            )

        smiles = self.drugs[drug_name]

        label = torch.tensor([row["LN_IC50"]], dtype=torch.float32)

        # Convert SMILES to latent vector
        x = get_representation([smiles], "FGR", self.fgroups_list, self.tokenizer)
        x = torch.tensor(x, dtype=torch.float32)
        print("Input vector shape:", x.shape)
        print("Expected encoder input size:", self.encoder[0].in_features)

        with torch.no_grad():
            z, _ = self.encoder(x)
        drug_vec = z.squeeze(0)

        try:
            omics_views = [
                torch.tensor(self.omics["expr"][cell_line], dtype=torch.float32),
                torch.tensor(self.omics["mut"][cell_line], dtype=torch.float32),
                torch.tensor(self.omics["meth"][cell_line], dtype=torch.float32),
                torch.tensor(self.omics["cnv"][cell_line], dtype=torch.float32),
            ]
        except KeyError as e:
            raise KeyError(f"Cell line '{cell_line}' missing from omics data: {e}")

        return drug_vec, omics_views, label
