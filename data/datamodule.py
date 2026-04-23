import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp 
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


def custom_collate_fn(batch):
    """Custom collate function to handle tensor and scalar targets properly."""
    uids = [item['uid'] for item in batch]
    sources = torch.stack([item['source'] for item in batch])  # [B, 3, H, W, D]
    targets = torch.tensor([item['target'] for item in batch], dtype=torch.long)  # [B]
    
    return {
        'uid': uids,
        'source': sources,
        'target': targets
    }


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 ds_train=None,
                 ds_val=None,
                 ds_test=None,
                 batch_size: int = 8,
                 batch_size_val: int = None,
                 batch_size_test: int = None,
                 num_train_samples: int = None,
                 num_workers: int = 0,  # Changed default to 0
                 seed: int = 42, 
                 pin_memory: bool = True,
                 weights: list = None,
                 prefetch_factor: int = 2,
                ):
        super().__init__()
        
        self.ds_train = ds_train 
        self.ds_val = ds_val 
        self.ds_test = ds_test 

        self.batch_size = batch_size
        self.batch_size_val = batch_size if batch_size_val is None else batch_size_val 
        self.batch_size_test = batch_size if batch_size_test is None else batch_size_test 
        self.num_train_samples = num_train_samples
        self.num_workers = min(num_workers, 2)  # Cap at 2 for memory safety
        self.seed = seed 
        self.pin_memory = pin_memory
        self.weights = weights
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        if self.ds_train is not None:
            if self.weights is not None:
                num_samples = len(self.weights) if self.num_train_samples is None else self.num_train_samples
                sampler = WeightedRandomSampler(self.weights, num_samples=num_samples, generator=generator) 
            else:
                num_samples = len(self.ds_train) if self.num_train_samples is None else self.num_train_samples
                sampler = RandomSampler(self.ds_train, num_samples=num_samples, replacement=False, generator=generator)
            
            loader_kwargs = {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "sampler": sampler,
                "generator": generator,
                "drop_last": True,
                "pin_memory": self.pin_memory,
                "collate_fn": custom_collate_fn,
                "timeout": 0,  # No timeout
            }
            
            # Add prefetch_factor only if num_workers > 0
            if self.num_workers > 0:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
                loader_kwargs["persistent_workers"] = True
            
            return DataLoader(self.ds_train, **loader_kwargs)
        
        raise AssertionError("A training set was not initialized.")

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        if self.ds_val is not None:
            loader_kwargs = {
                "batch_size": self.batch_size_val,
                "num_workers": 0,  # Always use main process for validation
                "shuffle": False,
                "generator": generator,
                "drop_last": False,
                "pin_memory": self.pin_memory,
                "collate_fn": custom_collate_fn,
            }
            
            return DataLoader(self.ds_val, **loader_kwargs)
        
        raise AssertionError("A validation set was not initialized.")

    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        if self.ds_test is not None:
            loader_kwargs = {
                "batch_size": self.batch_size_test,
                "num_workers": 0,  # Always use main process for testing
                "shuffle": False,
                "generator": generator,
                "drop_last": False,
                "pin_memory": self.pin_memory,
                "collate_fn": custom_collate_fn,
            }
            
            return DataLoader(self.ds_test, **loader_kwargs)
       
        raise AssertionError("A test set was not initialized.")
