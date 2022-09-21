
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import pandas
import numpy as np
from skimage import io
import os
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn import preprocessing
import albumentations as A
import cv2


def create_folds(df,n_fold = 5):
#     skf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=99)
#     for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['view'], groups = df["label"])):
#         df.loc[val_idx, 'fold'] = fold
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=99)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold

    return df


def get_transforms(train = True,cfg=None):
    img_size = [224,224]
    data_transforms = {
        "train": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=224//20, max_width=224//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        
        "valid": A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
    }
    if train==True:
        return data_transforms["train"] 
    else:
        return data_transforms['valid']

class PogChampDataset(Dataset):
    def __init__(self,
                 csv_path = '../input/kaggle-pog-series-s01e03/corn/train.csv',
                 train = True,
                 val = False,
                 fold = None,
                 transforms = None
        ):
        
        super().__init__();
        
        self.data_dir = '../input/kaggle-pog-series-s01e03/corn'
        
        self.csv = pd.read_csv(csv_path)
        
        if train or val:
            self.csv = create_folds(self.csv).reset_index()
            if fold is not None and train is True and val is False:
                self.csv = self.csv[self.csv['fold'] != fold].reset_index()
            if fold is not None and val is True and train is False:
                self.csv = self.csv[self.csv['fold'] == fold].reset_index()
            
            le = preprocessing.LabelEncoder()
            self.csv['label'] = le.fit_transform(self.csv.label.values)
        
        self.indexes = self.csv.index.values
        self.transforms = transforms
        self.train = train
        self.val = val
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.data_dir,self.csv['image'][idx])
#         print(self.csv['seed_id'][idx])
        img = io.imread(img_path)/255
        
        if self.transforms:
            data = self.transforms(image = img)
            img = data['image']
        
        img = np.transpose(img,(2,0,1))
        
        if not isinstance(img,torch.Tensor):
            img  = torch.tensor(img,dtype = torch.float32)
        
        if self.train is True or self.val is True:
            label = torch.tensor(int(self.csv['label'][idx]),dtype = torch.float32)
            return (img,label)
        else:
            return img
        
        
        
        
class PogChampDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "../input/kaggle-pog-series-s01e03",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        fold = 0
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )

        self.data_train: Optional[Dataset] = PogChampDataset(train = True,val = False,fold = self.hparams.fold,transforms = get_transforms(train = True))
        self.data_val: Optional[Dataset] =  PogChampDataset(train = False,val = True,fold = self.hparams.fold,transforms = get_transforms(train = False))
        self.data_test: Optional[Dataset] = PogChampDataset(train = False,
                                                               val = False,
                                                               csv_path = '../input/kaggle-pog-series-s01e03/corn/test.csv',
                                                               transforms = get_transforms(train = False)
                                                           )

    @property
    def num_classes(self):
        return int(self.data_train['label'].nunique())

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
#         MNIST(self.hparams.data_dir, train=True, download=True)
#         MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
#         if not self.data_train and not self.data_val and not self.data_test:
#             trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
#             testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
#             dataset = ConcatDataset(datasets=[trainset, testset])
#             self.data_train, self.data_val, self.data_test = random_split(
#                 dataset=dataset,
#                 lengths=self.hparams.train_val_test_split,
#                 generator=torch.Generator().manual_seed(42),
#             )
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
