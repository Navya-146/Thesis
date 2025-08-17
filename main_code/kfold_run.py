# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .venv\Scripts\Activate.ps1

import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import torch
import pandas as pd
from torch.utils.data import DataLoader

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from main_code.network import MultiViewNet, IC50LightningModel
from main_code.data_new import DrugOmicsIC50Dataset
from main_code.FGR.load_FGR import get_fgr_module, fgroups_list, tokenizer


# Configurations

SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-4
DROPOUT = 0.2
INPUT_SIZE = [256, 12798, 12798, 12798, 12798] # drug, expr, cnv, mut, meth
USE_OMICS = [True, False, False, False]
OUTPUT_SIZE = 1
LAYERS_BEFORE_COMB = [2048, 1024, 512, 256]
LAYERS_AFTER_COMB = [256, 64, 32]
COMB_TYPE = "concatenation" #or concatenation

PROJECT_PATH = "/home/da24c011/miniconda3/project/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
FGR_CKPT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "epoch_000_val_0.8505.ckpt")

#FGR_CKPT_PATH = r"C:\Users\pooja\OneDrive\Desktop\Documents\IITM\Thesis\Training data\final_ish\checkpoints\epoch_000_val_0.8505.ckpt"
PROJECT_NAME = "expr_concat" #omics_comb_cv_fold
CHECKPOINT_DIR = f"checkpoints_cv/{PROJECT_NAME}" 


RUN_NAME_BASE = "multiviewnet_cv_fold" 

#data 
full_df = pd.read_csv((os.path.join(PROJECT_PATH, "data to be used", "fin_ic50_21-7.csv")))

expression_df = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "expression_wo_cosmic.csv"))), index_col=0)
mutation_df = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "mutation_wo_cosmic.csv"))), index_col=0)
methylation_df = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "methylation_wo_mean_cosmic.csv"))), index_col=0)
cnv_df = pd.read_csv(((os.path.join(PROJECT_PATH, "raw data from colab", "cnv_wo_cosmic.csv"))), index_col=0)
gene_orders = {
    "exp": expression_df.columns.tolist(),
    "mut": mutation_df.columns.tolist(),
    "meth": methylation_df.columns.tolist(),
    "cnv": cnv_df.columns.tolist(),
}

omics_data = {
    "exp": expression_df.to_dict(orient="index"),
    "cnv": cnv_df.to_dict(orient="index"),
    "mut": mutation_df.to_dict(orient="index"),
    "meth": methylation_df.to_dict(orient="index"),   
}

drug_df = pd.read_csv((os.path.join(PROJECT_PATH, "data to be used", "drug_smiles.tsv")), sep="\t", index_col=0)
drug_dict = drug_df["SMILES"].to_dict()
print("data loaded!")
full_df = full_df[full_df['DRUG_NAME'].isin(drug_dict.keys())].reset_index(drop=True)

#encoder

fgr_encoder = get_fgr_module(FGR_CKPT_PATH)
encoder = fgr_encoder.net.encoder
print("encoder loaded!")

seed_everything(SEED)
train_val_df, test_df = train_test_split(
    full_df, test_size=0.2, stratify=full_df["TCGA cancer type"], random_state=SEED, shuffle=True
)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_strat = train_val_df["TCGA cancer type"]

# Training loop

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, y_strat)):
    print(f"\n--- Fold {fold + 1}/{NUM_FOLDS} ---")

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DrugOmicsIC50Dataset(train_df, fgr_encoder, omics_data, drug_dict, tokenizer, fgroups_list, INPUT_SIZE)
    val_dataset = DrugOmicsIC50Dataset(val_df, fgr_encoder, omics_data, drug_dict, tokenizer, fgroups_list, INPUT_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=60)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=40)

    # Initialize model 
    base_model = MultiViewNet(
        layers_before_comb=LAYERS_BEFORE_COMB,
        layers_after_comb=LAYERS_AFTER_COMB,
        use_omics=USE_OMICS,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        activation="relu",
        comb=COMB_TYPE,
        fgr_encoder = encoder
    )
    lightning_model = IC50LightningModel(base_model, lr=LR)

    # WandB logger
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=f"{RUN_NAME_BASE}_{fold + 1}",
        log_model=True,
    )

    wandb_logger.experiment.config.update({
        "fold": fold + 1,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LR,
    })

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/r2",
        mode="max",
        save_top_k=1,
        filename=f"fold{fold + 1}-best-r2-{{epoch:02d}}-{{val/r2:.4f}}",
        dirpath=os.path.join(CHECKPOINT_DIR, f"fold_{fold + 1}"),
        verbose=True,
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val/r2",
        patience=10,
        mode="max",
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu",
        devices=1,
        precision=16
    )

    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb_logger.experiment.finish()

    print(f"Best checkpoint for fold {fold + 1} saved at: {checkpoint_callback.best_model_path}")


# Final test set
print("\n--- Running final testing ---")


test_dataset = DrugOmicsIC50Dataset(test_df, fgr_encoder, omics_data, drug_dict, tokenizer, fgroups_list, gene_orders, INPUT_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=40)

# Load best checkpoint
best_checkpoint_path = checkpoint_callback.best_model_path
print(f"Loading model from checkpoint: {best_checkpoint_path}")


wandb_logger = WandbLogger(
    project=PROJECT_NAME,
    name="test",
    log_model=False,
)

wandb_logger.log_metrics({
    "test/r2": results[0]["test/r2"],
    "test/loss_mse": results[0]["test/loss_mse"],
    "test/adj_r2": results[0]["test/adj_r2"]
})


test_model_base = MultiViewNet(
    layers_before_comb=LAYERS_BEFORE_COMB,
    layers_after_comb=LAYERS_AFTER_COMB,
    use_omics=USE_OMICS,
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    dropout=DROPOUT,
    activation="relu",
    comb=COMB_TYPE,
    fgr_encoder = encoder
)

test_lightning_model = IC50LightningModel.load_from_checkpoint(best_checkpoint_path, model=test_model_base)
test_lightning_model.eval()

test_trainer = Trainer(accelerator="gpu", devices=1)

with torch.no_grad():
    results = test_trainer.test(model=test_lightning_model, dataloaders=test_loader)

print("Test results:", results)
wandb_logger.experiment.finish()
