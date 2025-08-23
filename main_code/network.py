from typing import List
import torch
from torch import nn
from main_code.FGR.load_FGR import get_fgr_model
from typing import List
from sklearn.metrics import r2_score
import torch
from torch import nn
from lightning import LightningModule



class MultiViewNet(nn.Module):
    """
    Neural network with:
      - pre-concat MLP per modality (drug projected + omics branches)
      - fusion (concatenation or attention)
      - post-concat MLP
    Behavior:
      - drug branch is always used
      - omics branches are created for each entry in input_size[1:]
      - use_omics masks which omics branches are active (True -> branch used)
      - input_size must have length == 1 + len(use_omics)
    """

    def __init__(
        self,
        layers_before_comb: List[int],
        layers_after_comb: List[int],
        use_omics: List[bool],               # e.g. [True, True, False, True]
        input_size: List[int],               # [drug_dim, exp_dim, cnv_dim, mut_dim, meth_dim]
        output_size: int,
        dropout: float = 0.2,
        activation: str = "relu",
        comb: str = "concatenation",   #or attention
        task: str = "regression",      #or classification
        fgr_encoder = nn.Module
    ):
        super().__init__()

        # validation
        if len(input_size) != 1 + len(use_omics):
            print(input_size, use_omics)
            raise ValueError(f"input_size must have length 1 + len(use_omics). Got {len(input_size)} vs {1 + len(use_omics)}")

        if len(layers_before_comb) < 1 or len(layers_after_comb) < 1:
            raise ValueError("layers_before_comb and layers_after_comb must each have at least 1 element.")

        if comb not in ("concatenation", "attention"):
            raise ValueError("comb must be 'concatenation' or 'attention'")

        self.layers_before_comb = layers_before_comb
        self.layers_after_comb = layers_after_comb
        self.use_omics = use_omics
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.comb = comb
        self.task = task
        self.fgr_encoder = fgr_encoder.eval()
        for param in self.fgr_encoder.parameters():
            param.requires_grad = False

        self.num_omics_enabled = sum(1 for u in use_omics if u)
        if self.num_omics_enabled < 1:
            raise ValueError("At least one omics modality must be enabled in 'use_omics'.")
        self.num_modalities = 1 + self.num_omics_enabled  # drug + active omics

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "celu":
            self.activation = nn.CELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        omics_input_sizes = input_size[1:]
        self.omics_branches = nn.ModuleList(
            [ self._make_pre_concat_block(in_dim, layers_before_comb) for in_dim in omics_input_sizes ]
        )

        # config
        if comb == "attention":
            # one weight per active modality
            self.attn_weights = nn.Parameter(torch.randn(self.num_modalities))
            fused_dim = layers_before_comb[-1]
        else:  # concatenation
            fused_dim = self.num_modalities * layers_before_comb[-1]
            
        self.postconcat = self._make_post_concat_block(fused_dim, layers_after_comb, dropout, self.activation)

    def _make_pre_concat_block(self, input_dim, layers_before_comb):
        """Create MLP that maps input_dim -> layers_before_comb[-1] using the sequence in layers_before_comb."""
        layers = []
        layers.append(nn.Linear(input_dim, layers_before_comb[0], dtype=torch.float))
        layers.append(nn.BatchNorm1d(layers_before_comb[0], dtype=torch.float))
        layers.append(self.activation)
        
        for i in range(len(layers_before_comb) - 2):
            layers.append(nn.Linear(layers_before_comb[i], layers_before_comb[i+1], dtype=torch.float))
            layers.append(nn.BatchNorm1d(layers_before_comb[i+1], dtype=torch.float))
            layers.append(self.activation)

        if len(layers_before_comb) >= 2:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(layers_before_comb[-2], layers_before_comb[-1], dtype=torch.float))
        return nn.Sequential(*layers)

    def _make_post_concat_block(self, input_dim, layers_after_comb, dropout, activation):
        """Create the MLP after concatenation/attention. Uses layers_after_comb list."""
        layers = []
        layers.append(nn.Linear(input_dim, layers_after_comb[0], dtype=torch.float))
        layers.append(nn.BatchNorm1d(layers_after_comb[0], dtype=torch.float))
        layers.append(activation)
        for i in range(len(layers_after_comb) - 2):
            layers.append(nn.Linear(layers_after_comb[i], layers_after_comb[i+1], dtype=torch.float))
            layers.append(nn.BatchNorm1d(layers_after_comb[i+1], dtype=torch.float))
            layers.append(activation)
        if len(layers_after_comb) >= 2:
            layers.append(nn.Linear(layers_after_comb[-2], layers_after_comb[-1], dtype=torch.float))
        layers.append(nn.Dropout(dropout))
        
        if self.task == "regression":
            layers.append(nn.Linear(layers_after_comb[-1], self.output_size, dtype=torch.float))
        elif self.task == "classification":
            layers.append(nn.Linear(layers_after_comb[-1], self.output_size, dtype=torch.float))
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        return nn.Sequential(*layers)

    def attention_fuse(self, views: List[torch.Tensor]) -> torch.Tensor:
        """
        Simple attention using a per-modality scalar weight (same across batch and features),
        matching previous design (softmax over the modality axis).
        views: list of tensors shape [batch, hidden]
        returns: [batch, hidden]
        """
        stacked = torch.stack(views, dim=1)  # [batch, num_active, hidden]
        attn = torch.softmax(self.attn_weights, dim=0)  # [num_modalities]
        # If some modalities are disabled, self.attn_weights still has size num_modalities (constructed at init)
        # and views length equals num_modalities; ordering should match (drug first, then enabled omics).
        weighted = stacked * attn.view(1, -1, 1)
        return weighted.sum(dim=1)

    def forward(self, drug_input: torch.Tensor, exp=None, cnv=None, mut=None, meth=None) -> torch.Tensor:
        """
        Accepts separate tensors for each omics modality (or None). Order: exp, cnv, mut, meth.
        The model only processes omics branches where use_omics[i] is True (the branch was created).
        The dataset may pass all tensors; model will ignore those for which use_omics is False.
        """
        omics_inputs = [exp, cnv, mut, meth]
        if len(omics_inputs) != len(self.omics_branches):
            raise RuntimeError("Unexpected number of omics inputs provided to forward()")
        
        # pre-concat processing
        views = []
        drug_proj = self.fgr_encoder(drug_input) 
        drug_proj = drug_proj.squeeze(1)
        views.append(drug_proj)

        # For each omics branch, only run it if use_omics[i] is True
        branch_idx = 0
        for i, (use_flag, branch) in enumerate(zip(self.use_omics, self.omics_branches)):
            if not use_flag:
                branch_idx += 1
                continue
            omics_tensor = omics_inputs[i]
            if omics_tensor is None:
                raise ValueError(f"Omics modality at index {i} is enabled in the model (use_omics={self.use_omics}), but forward received None for it.")
            processed = branch(omics_tensor)
            views.append(processed)
            branch_idx += 1

        if len(views) <= 1:
            raise ValueError("Model requires at least one omics modality in addition to drug_latent (check use_omics).")

        # fusion
        if self.comb == "concatenation":
            fused = torch.cat(views, dim=1)
        else:  
            fused = self.attention_fuse(views)

        # post-concat
        out = self.postconcat(fused)
        return out


class IC50LightningModel(LightningModule):
    def __init__(self, model: nn.Module, lr=1e-4, task = "regression"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.task = task
        #self.save_hyperparameters(ignore=["fgr_encoder", "tokenizer"])

        if task == "regression":
            self.loss_fn = nn.MSELoss()
        elif task == "classification":
            self.loss_fn = nn.BCEWithLogitsLoss()  # binary
        else:
            raise ValueError(f"Unsupported task: {task}")


    def forward(self, drug_vec, exp=None, cnv=None, mut=None, meth=None):
        return self.model(drug_vec, exp=exp, cnv=cnv, mut=mut, meth=meth)


    def _shared_step(self, batch, stage):
        drug_vec, exp, cnv, mut, meth, label = batch
        pred = self(drug_vec, exp, cnv, mut, meth)

        if self.task=="regression":
            pred = pred.squeeze(-1) 
            loss = self.loss_fn(pred, label)

            y_true = label.detach().cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            
            r2 = r2_score(y_true, y_pred)
            n = y_true.shape[0]
            omics_views = [exp, cnv, mut, meth]
            p = sum(tensor.shape[1] for tensor in omics_views if tensor is not None)
    
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    
    
            self.log(f"{stage}/loss_mse", loss, prog_bar=True)
            self.log(f"{stage}/r2", r2, prog_bar=True)
            self.log(f"{stage}/adj_r2", adj_r2, prog_bar=True)

        elif self.task=="classification":
            loss = self.loss_fn(pred, label.float().unsqueeze(1))
            probs = torch.sigmoid(pred)
            y_hat = (probs > 0.5).long()
            acc = (y_hat == label.view_as(y_hat)).float().mean()
            self.log(f"{stage}/loss_ce", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)

        else:
            raise ValueError(f"Unsupported Task: {self.task}")
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

