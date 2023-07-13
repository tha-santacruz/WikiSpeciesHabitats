## Standard library
import argparse
## Custom module
from execution import Executer
## Third party package
import torch
## Fixed random seed
torch.manual_seed(42)
## Use cuDNN
torch.backends.cudnn.benchmark = True

def parse_args():
    """
    Text-agnostic habitat classification model
    """
    parser = argparse.ArgumentParser(description="Hyperparameters and config parser")

    parser.add_argument("--BATCH_SIZE", dest="BATCH_SIZE",
                      help="{choose a batch size, prefferably a power of 2}",
                      default = 128,
                      type=int)
    
    parser.add_argument("--CHECKPOINTS_PATH", dest="CHECKPOINTS_PATH",
                      help="{Path to the checkpoints folder}",
                      default = "./../checkpoints/",
                      type=str)

    parser.add_argument("--DATASET_PATH", dest="DATASET_PATH",
                      help="{Path to the dataset}",
                      default = "./../final_data/",
                      type=str)

    parser.add_argument("--DEVICE", dest="DEVICE",
                      choices=["cuda:0", "cuda:1"],
                      help="{cuda device to use}",
                      default = "cuda:0",
                      type=str)
    
    parser.add_argument("--EMBEDDING", dest="EMBEDDING",
                      help="{agnostic, longformer or doc2vec}",
                      default = "longformer",
                      type=str)

    parser.add_argument("--EPOCH_NUM", dest="EPOCH_NUM",
                      help="{choose a number of epochs, -1 for infinite}",
                      default = 300,
                      type=int)
    
    parser.add_argument("--FRACTION", dest="FRACTION",
                      help="{fraction of train samples with species from same population pool as test (for progressive splitting)}",
                      default = 40,
                      type=int)
    
    parser.add_argument("--FUSION", dest="FUSION",
                      help="{sum prod mean or max}",
                      default = "sum",
                      type=str)
    
    parser.add_argument("--HIDDEN_SIZE", dest="HIDDEN_SIZE",
                      help="{choose a size for hidden layers}",
                      default = 8192,
                      type=int)
    
    parser.add_argument("--LEARNING_RATE", dest="LEARNING_RATE",
                      help="{choose a learning rate}",
                      default = 1e-5,
                      type=float)
    
    parser.add_argument("--LEVEL", dest="LEVEL",
                      help="{Ecosystem classification level (L1 or L2)}",
                      default = "L1",
                      type=str)

    parser.add_argument("--LOAD_CHECKPOINT", dest="LOAD_CHECKPOINT",
                      help="{whether of not to load a checkpoint}",
                      default = "no",
                      type = str)
    
    parser.add_argument("--MERGE_METHOD", dest="MERGE_METHOD",
                      help="{method used to merge text files}",
                      choices=["fusion","selection"],
                      default = "fusion",
                      type=str)
    
    parser.add_argument("--MIN_LEARNING_RATE", dest="MIN_LEARNING_RATE",
                      help="{choose a minimal learning rate}",
                      default = 1e-6,
                      type=float)
    
    parser.add_argument("--NORMALIZE", dest="NORMALIZE",
                      help="{normalize embedding vectors, yes or no}",
                      default = "yes",
                      type=str)
    
    parser.add_argument("--NUM_WORKERS", dest="NUM_WORKERS",
                      help="{Number of workers on CPU for data loading}",
                      default = 8,
                      type=int)
    
    parser.add_argument("--RANDOM_STATE", dest="RANDOM_STATE",
                      help="{Random state used to create the dataset}",
                      default = 1,
                      type=int)

    parser.add_argument("--RUN_MODE", dest="RUN_MODE",
                      choices=["train", "test"],
                      help="{train, test}",
                      default = "train",
                      type=str)
    
    parser.add_argument("--SAVE_CHECKPOINTS", dest="SAVE_CHECKPOINTS",
                      choices=["yes", "no"],
                      help="{yes, no}",
                      default = "yes",
                      type=str)
    
    parser.add_argument("--SPLITTING", dest="SPLITTING",
                      choices=["spatial", "species", "progressive"],
                      help="{data splitting criterion, can be spatial or species}",
                      default = "species",
                      type=str)

    parser.add_argument("--TARGET_BASE", dest="TARGET_BASE",
                      choices=["set", "species"],
                      help="{target creation base}",
                      default = "set",
                      type=str)
    
    parser.add_argument("--WANDB_ENTITY", dest="WANDB_ENTITY",
                      help="{wandb entity to save logs}",
                      default = "group_name",
                      type=str)
    
    parser.add_argument("--WANDB_LOGS", dest="WANDB_LOGS",
                      choices=["yes", "no"],
                      help="{Use of wandb logging}",
                      default = "yes",
                      type=str)

    parser.add_argument("--WANDB_PROJECT", dest="WANDB_PROJECT",
                      help="{wandb project to save logs}",
                      default = "fusion_tuning",
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
    else:
        print(f"Running job on GPU : {torch.cuda.get_device_name(cfg.DEVICE)}")
    print(cfg)
    ## Run (with or without optimized preprocessing)
    if cfg.RUN_MODE == "train":
        #print(cfg)
        execute = Executer(cfg)
        execute.train()
    elif cfg.RUN_MODE == "test":
        execute = Executer(cfg)
        execute.test()
