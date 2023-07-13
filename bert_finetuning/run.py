## Standard library
import argparse
import tempfile
## Custom module
from execution import Executer
## Third party package
import torch
## Use cuDNN
torch.backends.cudnn.benchmark = True

def parse_args():
    """
    DistilBert ecological info classification model
    """
    parser = argparse.ArgumentParser(description="Hyperparameters and config parser")

    parser.add_argument("--BATCH_SIZE", dest="BATCH_SIZE",
                      help="{choose a batch size, prefferably a power of 2}",
                      default = 1,
                      type=int)
    
    parser.add_argument("--CHECKPOINTS_PATH", dest="CHECKPOINTS_PATH",
                      help="{Path to the checkpoints folder}",
                      default = "./../checkpoints/",
                      type=str)

    parser.add_argument("--DATASET_PATH", dest="DATASET_PATH",
                      help="{Path to the dataset}",
                      default = "./",
                      type=str)

    parser.add_argument("--DEVICE", dest="DEVICE",
                      choices=["cuda:0", "cuda:1"],
                      help="{cuda device to use}",
                      default = "cuda:0",
                      type=str)

    parser.add_argument("--EPOCH_NUM", dest="EPOCH_NUM",
                      help="{choose a number of epochs, -1 for infinite}",
                      default = 100,
                      type=int)
    
    parser.add_argument("--LEARNING_RATE", dest="LEARNING_RATE",
                      help="{choose a learning rate}",
                      default = 1e-5,
                      type=float)
    
    parser.add_argument("--LOAD_CHECKPOINT", dest="LOAD_CHECKPOINT",
                      help="{.pth checkpoint file name ex : checkpoint.pth}",
                      default = None)
    
    parser.add_argument("--MIN_LEARNING_RATE", dest="MIN_LEARNING_RATE",
                      help="{choose a minimal learning rate}",
                      default = 1e-6,
                      type=float)
    
    parser.add_argument("--NUM_WORKERS", dest="NUM_WORKERS",
                      help="{Number of workers on CPU for data loading}",
                      default = 12,
                      type=int)

    parser.add_argument("--RUN_MODE", dest="RUN_MODE",
                      choices=["train", "test"],
                      help="{train, test}",
                      default = "train",
                      type=str)
    
    parser.add_argument("--SAVE_CHECKPOINTS", dest="SAVE_CHECKPOINTS",
                      choices=["yes", "no"],
                      help="{yes, no}",
                      default = "no",
                      type=str)
    
    parser.add_argument("--WANDB_ENTITY", dest="WANDB_ENTITY",
                      help="{wandb entity to save logs}",
                      default = "group_name",
                      type=str)
    
    parser.add_argument("--WANDB_LOGS", dest="WANDB_LOGS",
                      choices=["yes", "no"],
                      help="{Use of wandb logging}",
                      default = "no",
                      type=str)

    parser.add_argument("--WANDB_PROJECT", dest="WANDB_PROJECT",
                      help="{wandb project to save logs}",
                      default = "DistilBert_SLC",
                      type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    ## Configuration
    cfg = parse_args()

    ## Use CPU if no GPU is available
    if not torch.cuda.is_available():
        cfg.DEVICE = "cpu"
        print("No GPU found, using CPU")

    ## Run (with or without optimized preprocessing)
    if cfg.RUN_MODE == "train":
        #print(cfg)
        execute = Executer(cfg)
        execute.train()
    if cfg.RUN_MODE == "test":
        execute = Executer(cfg)
        execute.test()
