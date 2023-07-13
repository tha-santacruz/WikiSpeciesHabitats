import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import PIL
from time import time
import math
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from dataset import BertTuningDataset
from utils import EarlyStopper, ClassWiseConfusionMatrix


class Executer():
    def __init__(self, cfg):
        """Initialization"""
        ## Config
        self.cfg = cfg
        ## Preprocessing
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.preprocessing = lambda x : tokenizer(x, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        ## Datasets
        self.train_set = BertTuningDataset(path=self.cfg.DATASET_PATH, split="train", preprocessing=self.preprocessing)
        self.val_set = BertTuningDataset(path=self.cfg.DATASET_PATH, split="val", preprocessing=self.preprocessing)
        ## Network
        self.net = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device=self.cfg.DEVICE)
        ## Model weights setting
        if self.cfg.LOAD_CHECKPOINT:
            path = self.cfg.CHECKPOINTS_PATH + self.cfg.LOAD_CHECKPOINT
            self.net.load_state_dict(torch.load(path))
            print(f"Loaded model state dict from file {path}")
        ## Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.LEARNING_RATE)
        fac = 1/math.pow(10,1/3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=fac, patience=3, cooldown=0, threshold=1e-2, min_lr=self.cfg.MIN_LEARNING_RATE)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.early_stopper = EarlyStopper(patience=10, minimize=True)
        ## Loss (NB : CrossEntropyLoss is softmax AND cross entropy)
        self.criterion = nn.CrossEntropyLoss()

    def plot_confusion_matrix(self, cf_matrix, save_image=False):
        """Creation of an image out of the confusion matrix"""
        ## Plotging
        plt.figure(figsize=(5,4))
        ax = sns.heatmap(cf_matrix, annot=True, fmt='.0f', annot_kws={"size":16}, cmap='Greens', cbar=True) 
                        #xticklabels=self.train_set.classes_codes, yticklabels=self.train_set.classes_names)
        ax.set(xlabel="Predicted", ylabel="True")
        ax.xaxis.tick_top()
        plt.rc('axes', labelsize=30)
        plt.tight_layout()
        if save_image==True:
            plt.savefig("confusion_matrix.png")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        image = PIL.Image.open(buffer)
        plt.close() 
        return image

    def compute_accuracy(self, pred, target):
        """Overall and classwise accuracy computing"""
        ## Answers comparison
        comp = pred.argmax(dim=1)==target.argmax(dim=1) # batch overall accuracy : comp.sum()/comp.size(0)
        ## Get count of classes
        classes_instances = target.sum(dim=0)
        ## Get count of well predicted classes
        classes_goodpreds = torch.mul(target,comp[:,None]).sum(dim=0)
        return classes_instances.cpu().detach(), classes_goodpreds.cpu().detach()

    def save_net_params(self, epoch):
        """Model parameters checkpoint"""
        name = f"/distilbert_checkpoint_epoch_{epoch+1}.pth"
        #name = f"/bestmodel_{self.cfg.EMBEDDING}_{self.cfg.LEVEL}_classes.pth"
        torch.save(self.net.state_dict(), self.cfg.CHECKPOINTS_PATH + name)
        
    def infer_round(self, dataloader):
        """Forward pass and metrics computation"""
        ## Set model to eval (no dropout, batchnorm fits number of samples, etc..)
        self.net.eval()
        ## Initialize metrics
        acc_loss = 0
        acc_instances = torch.zeros(2)
        acc_goodpreds = torch.zeros(2)
        step_count = 0
        cf_matrix = torch.zeros((2,2)).numpy()
        for minibatch in dataloader:
            ## Predict batch
            input = minibatch[0].to(dtype=torch.int64, device=self.cfg.DEVICE, non_blocking=True)
            mask = minibatch[1].to(dtype=torch.int64, device=self.cfg.DEVICE, non_blocking=True)
            target = minibatch[2].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            pred = self.net(input_ids = input, attention_mask = mask).logits
            # Confusion matrix
            cf_matrix += confusion_matrix(target.argmax(dim=1).cpu(),pred.argmax(dim=1).cpu(), labels = (torch.arange(2)).numpy())
            ## Update metrics
            loss = self.criterion(pred, target)
            acc_loss += loss.item()
            classes_instances, classes_goodpreds = self.compute_accuracy(pred, target)
            acc_instances += classes_instances
            acc_goodpreds += classes_goodpreds
            step_count += 1
        cf_matrix = self.plot_confusion_matrix(cf_matrix, save_image=True)

        return acc_loss/step_count, acc_goodpreds.sum()/acc_instances.sum(), torch.nan_to_num(acc_goodpreds/acc_instances), cf_matrix
    
    def train_round(self, dataloader):
        """Forward pass, backard pass, metrics computation and parameters update"""
        ## Set model to train mode
        self.net.train()
        ## Initialize metrics
        acc_loss = 0
        acc_instances = torch.zeros(2)
        acc_goodpreds = torch.zeros(2)
        step_count = 0
        for minibatch in dataloader:
            ## Reset gradients
            self.optimizer.zero_grad()
            ## Predict batch
            input = minibatch[0].to(dtype=torch.int64, device=self.cfg.DEVICE, non_blocking=True)
            mask = minibatch[1].to(dtype=torch.int64, device=self.cfg.DEVICE, non_blocking=True)
            target = minibatch[2].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            pred = self.net(input_ids = input, attention_mask = mask).logits
            ## Backpropagate loss and update parameters
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            ## Metrics update
            acc_loss += loss.item()
            classes_instances, classes_goodpreds = self.compute_accuracy(pred, target)
            acc_instances += classes_instances
            acc_goodpreds += classes_goodpreds
            step_count += 1
        return acc_loss/step_count, acc_goodpreds.sum()/acc_instances.sum(), torch.nan_to_num(acc_goodpreds/acc_instances)

    def train(self):
        """Training loop"""
        ## Prepare for training
        train_loader = DataLoader(self.train_set, shuffle=True, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(self.val_set, shuffle=True, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
        start_time = time()
        ## Setting performance visualisation
        if self.cfg.WANDB_LOGS == "yes":
            experiment = wandb.init(project=self.cfg.WANDB_PROJECT, entity=self.cfg.WANDB_ENTITY)
            experiment.config.update(self.cfg)
        pbar = trange(self.cfg.EPOCH_NUM)
        for epoch in pbar:
            ## Resample negative examples
            self.train_set.resample_negatives()
            ## Train
            train_loss, train_overall_acc, train_classwise_acc = self.train_round(train_loader)
            ## Validate
            val_loss, val_overall_acc, val_classwise_acc, cf_matrix = self.infer_round(val_loader)
            ## Log progess
            pbar.set_description(f"train/val metrics at epoch {epoch+1} | loss : {train_loss:.2f}/{val_loss:.2f} | acc : {train_overall_acc:.2f}/{val_overall_acc:.2f}")
            #input("continue?")
            ## Save model state
            if self.cfg.SAVE_CHECKPOINTS == "yes":
                self.save_net_params(epoch)
            ## Transfer logs to wandb
            if self.cfg.WANDB_LOGS == "yes":
                ## Create logs dict
                logs_dict = {
                    "epoch": epoch+1,
                    "time": time()-start_time,
                    "learning rate": self.optimizer.param_groups[0]['lr'],
                    "train loss": train_loss,
                    "train overall accuracy": train_overall_acc,
                    "train class 0 accuracy": train_classwise_acc[0],
                    "train class 1 accuracy": train_classwise_acc[1],
                    "val loss": val_loss,
                    "val overall accuracy": val_overall_acc,
                    "val class 0 accuracy": val_classwise_acc[0],
                    "val class 1 accuracy": val_classwise_acc[1],
                    "confusion matrix": wandb.Image(cf_matrix)
                    }
                #print(logs_dict)
                experiment.log(logs_dict)

    def test(self):
        """Testing loop"""
        ## Prepare for test
        val_loader = DataLoader(self.val_set, shuffle=True, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
        ## Setting performance visualisation
        if self.cfg.WANDB_LOGS == "yes":
            experiment = wandb.init(project=self.cfg.WANDB_PROJECT, entity=self.cfg.WANDB_ENTITY)
            experiment.config.update(self.cfg)
        val_loss, val_overall_acc, val_classwise_acc, cf_matrix = self.infer_round(val_loader)
            
            

