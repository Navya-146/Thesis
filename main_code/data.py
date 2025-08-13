from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# import torch
from lightning.pytorch import LightningDataModule
from pandas.core.api import Series as Series
from rich.console import Console
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

console = Console()
ArrayLike = Union[List, tuple, np.ndarray]
DataFrameLike = Union[pd.DataFrame, pl.DataFrame]


import pandas as pd
import numpy as np

expression_df = pd.read_csv("final_ish/raw data from colab/expression_wo_cosmic.csv", index_col=0)
mutation_df = pd.read_csv("final_ish/raw data from colab/mutation_wo_cosmic.csv", index_col=0)
methylation_df = pd.read_csv("final_ish/raw data from colab/methylation_wo_cosmic.csv", index_col=0)
cnv_df = pd.read_csv("final_ish/raw data from colab/cnv_wo_cosmic.csv", index_col=0)

expression = expression_df.to_dict(orient="index")
mutation = mutation_df.to_dict(orient="index")
methylation = methylation_df.to_dict(orient="index")
cnv = cnv_df.to_dict(orient="index")

omics_data = {
    'expr': expression,
    'mut': mutation,
    'meth': methylation,
    'cnv': cnv,
}



def read_data(path: Union[Path, str], format: str = "parquet") -> pd.DataFrame:
    """Reads the data from the given path.
    Needs three data: Observations, Genotype and Latent data

    Args:
        path (Union[Path, str]): The path to the observation data.
        format (str, optional): The format of the data. Defaults to "parquet".

    Returns:
        pd.DataFrame: The data read from the path.
    """
    match format:
        case "feather":
            return pd.read_feather(path)
        case "parquet":
            return pd.read_parquet(path)
        case "csv":
            return pd.read_csv(path)
        case _:
            raise NotImplementedError(
                "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
            )


class EncodeDataset(Dataset):
    """Dataset used by Pytorch.

    Args:
        X (np.array): Input Variable
        y (np.array): Target Variable
        labels (Iterable, optional): Labels for the dataset. Defaults to None.
    """

    def __init__(self, X, y, labels: Iterable = None) -> None:
        assert len(X) == len(y), "X, y must have the same length."
        # assert X.dtype == np.float32 and y.dtype == np.float32, "X, y must be of type np.float32."

        self.X = X
        self.y = y

        self.size = len(X)
        self.labels = labels

    def __len__(self):
        """Returns the length of the object.

        :return: The length of the object.
        :rtype: int
        """
        return self.size

    def __getitem__(self, index) -> Any:
        """Get the item at the given index from the X and y arrays.

        Args:
            index (int): Index of the item to get.

        Returns:
            tuple: A tuple of two elements.
        """
        return self.X[index].astype(np.float32), np.asarray(self.y[index]).reshape(1)


class KFoldEncodeModule(LightningDataModule):
    """Lightning data module for K-Fold Cross Validation.

    Args:
        path (str): path to the dataset
        format (str, optional): dataset format. Defaults to None.
        k (int, optional): index of the k-fold. Defaults to 0.
        split_seed (int, optional): Random seed. Defaults to 42.
        num_splits (int, optional): Number of Splits. Defaults to 5.
        num_workers (int, optional): Number of workers. Defaults to 4.
        batch_size (int, optional): batch size. Defaults to 64.
        test_size (float, optional): test size. Defaults to 0.2.
        stratify (str, optional): stratify column. Defaults to None.
    """

    def __init__(
        self,
        path: str,
        format: str = None,
        k: int = 0,
        split_seed: int = 42,
        num_splits: int = 5,
        num_workers: int = 4,
        batch_size: int = 64,
        test_size: float = 0.2,
        stratify: "str" = None,
    ):
        super().__init__()
        self.path = path

        if format is None:
            self.format = path.split(".")[-1]
        else:
            self.format = format

        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size
        self.stratify = stratify

        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage=None):
        """Set up the data for the experiment.

        Args:
            stage (optional): The stage of the setup process. Defaults to None.

        Raises:
            NotImplementedError: If the file format is not supported.

        Returns:
            None
        """
        # print("Setting up data")

        match self.format:
            case "feather":
                data = pd.read_feather(self.path)
            case "parquet":
                data = pd.read_parquet(self.path)
            case "csv":
                data = pd.read_csv(self.path)
            case _:
                raise NotImplementedError(
                    "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
                )

        self.X = data.drop(
            ["Phenotype", "Condition", "Strain"], axis=1
        )  # TODO: Need to generalize
        self.y = data["Phenotype"].astype(np.float32)
        self.stratify_col = data[self.stratify] if self.stratify else None

        try:
            self.condition = data["Condition"]
        except:  # noqa: E722
            self.condition = None

        self.dataset = EncodeDataset(self.X.values, self.y.values)

        del self.X, self.y, data

        train_indices, test_indices = train_test_split(
            np.arange(len(self.dataset)),
            test_size=self.test_size,
            random_state=self.split_seed,
            stratify=self.stratify_col,
        )

        # Create test and training datasets
        self.test_dataset = Subset(self.dataset, test_indices)
        self.dataset_for_split = Subset(self.dataset, train_indices)

        if not self.train_dataset and not self.val_dataset:
            # Create KFold object
            kf = KFold(
                n_splits=self.num_splits, shuffle=True, random_state=self.split_seed
            )

            # Split data into training and validation subsets
            all_splits = list(kf.split(self.dataset_for_split))
            train_index, val_index = all_splits[self.k]

            # Create training and validation datasets
            self.train_dataset = Subset(self.dataset_for_split, train_index)
            self.val_dataset = Subset(self.dataset_for_split, val_index)

    def train_dataloader(self):
        """Returns a PyTorch DataLoader object for the training dataset.

        Parameters:
            None

        Returns:
            DataLoader: The DataLoader object for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Generate a validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Generate a test dataloader.

        Returns:
            DataLoader: The DataLoader object created with the specified parameters.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


class EncodeModule(LightningDataModule):
    """Lightning data module for K-Fold Cross Validation.

    Args:
        path (str): path to the dataset
        format (str, optional): dataset format. Defaults to "feather".
        k (int, optional): index of the k-fold. Defaults to 0.
        split_seed (int, optional): Random seed. Defaults to 42.
        num_splits (int, optional): Number of Splits. Defaults to 5.
        num_workers (int, optional): Number of workers. Defaults to 4.
        batch_size (int, optional): batch size. Defaults to 64.
        test_size (float, optional): test size. Defaults to 0.2.
    """

    def __init__(
        self,
        path: str,
        format: str = "feather",
        split_seed: int = 42,
        num_workers: int = 4,
        batch_size: int = 64,
        test_size: float = 0.2,
    ):
        super().__init__()

        self.path = path

        if format is None:
            self.format = path.split(".")[-1]
        else:
            self.format = format

        self.split_seed = split_seed
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_size = test_size

    def setup(self, stage=None):
        """Sets up the data for the model.

        Parameters:
            stage (str, optional): The stage of the setup process. Defaults to None.

        Raises:
            NotImplementedError: If the file format is not supported.

        Returns:
            None
        """
        print("Setting up data")

        match self.format:
            case "feather":
                data = pd.read_feather(self.path)
            case "parquet":
                data = pd.read_parquet(self.path)
            case "csv":
                data = pd.read_csv(self.path)
            case _:
                raise NotImplementedError(
                    "File format not supported. Most be one of 'feather', 'parquet', 'csv'"
                )

        self.X = data.drop(["Phenotype", "Condition", "Strain"], axis=1)
        self.y = data["Phenotype"].astype(np.float32)

        try:
            self.condition = data["Condition"]
        except:  # noqa: E722
            self.condition = None

        # Split data into test and training subsets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.split_seed,
        )

        self.test_dataset = EncodeDataset(X_val.values, y_val.values)

        del X_val, y_val

        # Split training subset into training and validation subsets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.test_size,
            random_state=self.split_seed,
        )

        self.train_dataset = EncodeDataset(X_train.values, y_train.values)
        self.val_dataset = EncodeDataset(X_val.values, y_val.values)

        del X_train, X_val, y_train, y_val, self.X, self.y

    def train_dataloader(self):
        """Generates a training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Generates a validation data loader.

        Returns:
            DataLoader: A data loader object.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create and return a DataLoader object for the test dataset.

        Returns:
            A DataLoader object for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )


class CancerDataset(Dataset):
    """Dataset for the Cancer data - PRISM_19Q4.

    args:
        X (DataFrameLike): Genotype data
        y (ArrayLike): Phenotype
        geno_data (DataFrameLike, optional): Genotype data. Defaults to None.
        latent_data (DataFrameLike, optional): Latent data. Defaults to None.
        stage (str, optional): The stage of the setup process. Defaults to None. Only checks for prediction stage

    """

    def __init__(
        self,
        X: DataFrameLike,
        y: Union[ArrayLike, pd.Series],
        geno_data: pd.DataFrame = None,
        latent_data: pd.DataFrame = None,
        stage=None,
    ) -> None:
        self.strain, self.condition = (
            X[X.columns[0]].values,
            X[X.columns[1]].values,
        )  # Contains information on the observations - should be of the form: [cell_line, condition]
        self.y = y.astype(np.float32)  # Phenotype
        self.geno_data = geno_data.astype(
            np.float32
        )  # Genotype data - Index should contain the strain information
        self.latent_data = (
            latent_data  # Latent data - Index should contain the condition information
        ).astype(np.float32)
        self.stage = stage

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        strains, conditions = self.strain[idx], self.condition[idx]

        if (self.geno_data is not None) and (self.latent_data is not None):
            if self.stage == "predict":
                return np.hstack(
                    (
                        self.geno_data.loc[strains].values,
                        self.latent_data.loc[conditions].values,
                    )
                )

            return np.hstack(
                (
                    self.geno_data.loc[strains].values,
                    self.latent_data.loc[conditions].values,
                )
            ), self.y[idx].reshape(
                1,
            )

        elif (self.geno_data is not None) and (self.latent_data is None):
            if self.stage == "predict":
                return self.geno_data.loc[strains].values
            return self.geno_data.loc[strains].values, self.y[idx].reshape(1)

        elif (self.geno_data is None) and (self.latent_data is not None):
            if self.stage == "predict":
                return self.latent_data.loc[conditions].values
            return self.latent_data.loc[conditions].values, self.y[idx].reshape(1)

        else:
            return self.y[idx].reshape(1)


class FGRCancerDataset(CancerDataset):
    def __init__(
        self,
        X: DataFrameLike,
        y: ArrayLike | Series,
        geno_data: pd.DataFrame = None,
        latent_data: pd.DataFrame = None,
        stage=None,
    ) -> None:
        super().__init__(X, y, geno_data, latent_data, stage)

    def __getitem__(self, idx):
        """Only difference is here we return x as a tuple of two elements.

        Args:
            idx (_type_): index
        """
        strains, conditions = self.strain[idx], self.condition[idx]

        if (self.geno_data is not None) and (self.latent_data is not None):
            if self.stage == "predict":
                return (
                    self.geno_data.loc[strains].values,
                    self.latent_data.loc[conditions].values,
                )

            return (
                self.geno_data.loc[strains].values,
                self.latent_data.loc[conditions].values,
            ), self.y[idx].reshape(
                1,
            )

        elif (self.geno_data is not None) and (self.latent_data is None):
            if self.stage == "predict":
                return self.geno_data.loc[strains].values
            return self.geno_data.loc[strains].values, self.y[idx].reshape(1)

        elif (self.geno_data is None) and (self.latent_data is not None):
            if self.stage == "predict":
                return self.latent_data.loc[conditions].values
            return self.latent_data.loc[conditions].values, self.y[idx].reshape(1)

        else:
            return self.y[idx].reshape(1)


class CancerKFoldModule(LightningDataModule):
    """Lightning data module for the  Cancer Dataset - PRISM_19Q4.

    Initializes the object with the given parameters.

    Args:
        path (List[str]): The path to the data files folder.
        presplit (bool, optional): Indicates if the data is already split. Defaults to True.
        latent_path (Union[Path, str], optional): The path to the latent data. Defaults to None, i.e. same as `path`.
        geno_path (Union[Path, str], optional): The path to the geno data. Defaults to None i.e. same as `path`.
        format (str, optional): The format of the data files. Defaults to "parquet".
        k (int, optional): The value of k. Defaults to 0.
        split_seed (int, optional): The seed for splitting the data. Defaults to 42.
        num_splits (int, optional): The number of data splits. Defaults to 5.
        test_size (float, optional): The test size. Defaults to 0.2. Will not be used if data is already split.
        num_workers (int, optional): The number of workers. Defaults to 4.
        batch_size (int, optional): The batch size. Defaults to 64.
        use_geno_data (bool, optional): Whether to use geno data. Defaults to True.
        use_latent_data (bool, optional): Whether to use latent data. Defaults to True.
        fgr (bool, optional): Whether to use FGR. Defaults to False.

        Both use_latent_data and use_geno_data cannot be False at the same time.
    """

    def __init__(
        self,
        path: List[str],
        presplit: bool = True,
        latent_path: Union[Path, str] = None,
        geno_path: Union[Path, str] = None,
        format: str = "parquet",
        k: int = 0,
        split_seed: int = 42,
        num_splits: int = 5,
        test_size: float = 0.2,
        num_workers: int = 4,
        batch_size: int = 64,
        use_geno_data: bool = True,
        use_latent_data: bool = True,
        fgr: bool = False,
    ):
        super().__init__()

        self.split_seed = split_seed
        self.num_workers = num_workers
        self.k = k
        self.num_splits = num_splits
        self.test_size = test_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.path = Path(path)
        self.presplit = presplit

        # Save paths
        if latent_path is not None:
            self.latent_path = Path(latent_path)
            # print(self.path)
        else:
            if self.path.is_dir():
                self.latent_path = self.path / f"latent.{format}"
            else:
                raise ValueError(f"latent_path: {self.latent_path} is not a directory")

        # console.log(self.path.is_dir())
        if geno_path is not None:
            self.geno_path = Path(geno_path)
        else:
            if self.path.is_dir():
                self.geno_path = self.path / f"genotype.{format}"
            else:
                raise ValueError(f"latent_path: {self.geno_path} is not a directory")
        # print(self.path)
        self.format = format

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        if (use_geno_data is False) and (use_latent_data is False):
            raise ValueError(
                "Both use_latent_data and use_geno_data cannot be False at the same time"
            )

        self.use_geno_data = use_geno_data
        self.use_latent_data = use_latent_data

        if fgr:
            self.fgr = fgr
            self.data_class = FGRCancerDataset
        else:
            self.fgr = False
            self.data_class = CancerDataset

        self.save_hyperparameters()

    def setup(self, stage=None):
        """Sets up the data for the model.

        Parameters:
            stage (str, optional): The stage of the setup process. Defaults to None.

        Returns:
            None
        """
        console.log("Setting up data")

        if self.use_geno_data and self.use_latent_data:
            latent_data = read_data(self.latent_path).set_index("Condition")
            geno_data = read_data(self.geno_path).set_index("cell_lines")
        elif self.use_geno_data and not self.use_latent_data:
            geno_data = read_data(self.geno_path).set_index("cell_lines")
            latent_data = None
        elif not self.use_geno_data and self.use_latent_data:
            latent_data = read_data(self.latent_path).set_index("Condition")
            geno_data = None

        x_col = ["cell_lines", "Condition"]
        y_col = "Phenotype"

        # if the data is already split into test and train, the files must be named train and test or else the file should be named `data`
        if not self.presplit:
            data_size = pa.dataset.dataset(self.path / "data.parquet").count_rows()

            console.log("Splitting data")
            train_indices, test_indices = train_test_split(
                np.arange(data_size),
                test_size=self.test_size,
                random_state=self.split_seed,
            )

        if stage == "test":
            if not self.presplit:
                self.test_df = pq.read_table(self.path / "data.parquet").take(
                    test_indices
                )

            else:
                self.test_df = pq.read_table(self.path / "test.parquet")

            self.test_dataset = self.data_class(
                self.test_df.to_pandas()[x_col],
                self.test_df.to_pandas()[y_col],
                geno_data=geno_data,
                latent_data=latent_data,
            )
            del self.test_df

        elif stage == "fit":
            if not self.presplit:
                self.train_df = pq.read_table(self.path / "data.parquet").take(
                    train_indices
                )
            # console.log("Reading Training data")
            else:
                self.train_df = pl.read_parquet(self.path / "train.parquet")
            # console.log("Read Training data")

            # console.log('Preparing dataset for KFold')
            self.dataset_for_split = self.data_class(
                self.train_df.to_pandas()[x_col],
                self.train_df.to_pandas()[y_col],
                geno_data=geno_data,
                latent_data=latent_data,
            )
            del self.train_df

            # console.log("Generating KFold splits")
            # KFold validation data generation
            if not self.train_dataset and not self.val_dataset:
                # Create KFold object
                kf = KFold(
                    n_splits=self.num_splits,
                    shuffle=True,
                    random_state=self.split_seed,
                )

                # Generate KFold splits
                all_splits = list(kf.split(self.dataset_for_split))
                train_index, val_index = all_splits[
                    self.k
                ]  # Get indices for particular fold

                # Create training and validation subsets
                self.train_dataset = Subset(self.dataset_for_split, train_index)
                self.val_dataset = Subset(self.dataset_for_split, val_index)

                del self.dataset_for_split

    def train_dataloader(self):
        """Generates a training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Generate a validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Generate a test dataloader.

        Returns:
            DataLoader: The DataLoader object created with the specified parameters.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    # Testing

    # dm1 = KFoldEncodeModule(
    #     path="/storage/bt20d204/data/bloom2013_shared_clf_3_pubchem.feather",
    # )
    # dm1.setup()

    # dm2 = KFoldEncodeModule(
    #     path="/storage/bt20d204/data/bloom2013_shared_clf_3_pubchem.feather",
    #     k=1,
    #     stratify="Phenotype",
    # )
    # dm2.setup()

    # loader = dm1.val_dataloader()
    # X, y = next(iter(loader))

    # loader2 = dm2.val_dataloader()

    # X2, y2 = next(iter(loader2))
    # print(X, "\n", X2)

    # print(torch.eq(X, X2).all().item()), "X and X2 is different"

    dm3 = CancerKFoldModule(
        path="/home/rajeeva/Project/data/cancer/PRISM_19Q4/",
        # geno_path="/home/rajeeva/Project/data/cancer/PRISM_19Q4_classification/genotype.parquet",
        latent_path="/home/rajeeva/Project/data/cancer/ChemGroup.parquet",
        batch_size=64,
        num_workers=10,
        fgr=True,
    )
    dm3.setup(stage="fit")

    for X, y in dm3.val_dataloader():
        print(len(X), y.shape, end="\t")
    pass
