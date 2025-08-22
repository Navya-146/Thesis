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
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 1e-4
DROPOUT = 0.2
INPUT_SIZE = [256, 12798, 12798, 12798, 12798] # drug, expr, cnv, mut, meth
USE_OMICS = [False, False, True, False]
OUTPUT_SIZE = 1
LAYERS_BEFORE_COMB = [1024,512, 256] #random 
LAYERS_AFTER_COMB = [256,64,16]     #random
COMB_TYPE = "concatenation" #or attention
TASK = "regression" #or classification
THRESHOLD = 0.0 #specify median for classification task

if TASK == "regression":
    monitor_metric = "val/r2"
elif TASK == "classification":
    monitor_metric = "val/accuracy" 
else:
    raise ValueError(f"Unknown TASK {TASK}")


PROJECT_PATH = "/home/da24c011/miniconda3/project/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
FGR_CKPT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "epoch_000_val_0.8505.ckpt")

#FGR_CKPT_PATH = r"C:\Users\pooja\OneDrive\Desktop\Documents\IITM\Thesis\Training data\final_ish\checkpoints\epoch_000_val_0.8505.ckpt"
PROJECT_NAME = f"mutation_{COMB_TYPE[:6]}_{TASK[:7]}" #omics_comb_task
CHECKPOINT_DIR = f"checkpoints_cv/{PROJECT_NAME}" 
RUN_NAME_BASE = "cv_fold" 

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

    train_dataset = DrugOmicsIC50Dataset(train_df, fgr_encoder, omics_data, drug_dict, tokenizer, fgroups_list, INPUT_SIZE, task=TASK, ic50_threshold=THRESHOLD)
    val_dataset = DrugOmicsIC50Dataset(val_df, fgr_encoder, omics_data, drug_dict, tokenizer, fgroups_list, INPUT_SIZE, task=TASK, ic50_threshold=THRESHOLD)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

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
        fgr_encoder = encoder,
        task = TASK
    )
    lightning_model = IC50LightningModel(base_model, lr=LR, task=TASK)

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
        monitor = monitor_metric,
        mode="max",
        save_top_k=1,
        filename=f"fold{fold + 1}-best-{monitor_metric}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        dirpath=os.path.join(CHECKPOINT_DIR, f"fold_{fold + 1}"),
        verbose=True,
    )

    # Early stopping callback
    #early_stop_callback = EarlyStopping(monitor=monitor_metric,patience=10,mode="max",verbose=True,)
    #not to be set for now
    
    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        precision = "16-mixed"
    )

    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wandb_logger.experiment.finish()
    print(f"Best checkpoint for fold {fold + 1} saved at: {checkpoint_callback.best_model_path}")


# Final test set
print("\n--- Running final testing ---")


test_dataset = DrugOmicsIC50Dataset(test_df, fgr_encoder, omics_data, drug_dict, tokenizer, fgroups_list, INPUT_SIZE,  task=TASK, ic50_threshold=THRESHOLD)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# Load best checkpoint
best_checkpoint_path = checkpoint_callback.best_model_path
print(f"Loading model from checkpoint: {best_checkpoint_path}")

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

test_trainer = Trainer(accelerator="gpu", devices=1,)#precision not required for testing

with torch.no_grad():
    results = test_trainer.test(model=test_lightning_model, dataloaders=test_loader)

wandb_logger = WandbLogger(
    project=PROJECT_NAME,
    name="test",
    log_model=False,
)

if TASK == "regression":
    wandb_logger.log_metrics({
        "test/r2": results[0]["test/r2"],
        "test/loss_mse": results[0]["test/loss_mse"],
        "test/adj_r2": results[0]["test/adj_r2"]
    })
elif TASK == "classification":
    wandb_logger.log_metrics({
        "test/accuracy": results[0]["test/accuracy"],
    })

print("Test results:", results)
wandb_logger.experiment.finish()
