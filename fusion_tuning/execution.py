import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from model import MLP
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import PIL
from time import time
import math
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelHammingDistance

from dataset import WikiSpeciesHabitats
from utils import EarlyStopper, ClassWiseConfusionMatrix


class Executer():
    def __init__(self, cfg):
        """Initialization"""
        ## Config
        self.cfg = cfg
        ## Datasets
        self.train_set = WikiSpeciesHabitats(path=self.cfg.DATASET_PATH, split="train", merge_method=self.cfg.MERGE_METHOD, random_state=self.cfg.RANDOM_STATE, 
                                            target_base=self.cfg.TARGET_BASE, splitting=self.cfg.SPLITTING, fraction=self.cfg.FRACTION, embedding=self.cfg.EMBEDDING, 
                                            level=self.cfg.LEVEL, fusion=self.cfg.FUSION, normalize=self.cfg.NORMALIZE)
        self.val_set = WikiSpeciesHabitats(path=self.cfg.DATASET_PATH, split="val", merge_method=self.cfg.MERGE_METHOD, random_state=self.cfg.RANDOM_STATE, 
                                            target_base=self.cfg.TARGET_BASE, splitting=self.cfg.SPLITTING, fraction=self.cfg.FRACTION, embedding=self.cfg.EMBEDDING, 
                                            level=self.cfg.LEVEL, fusion=self.cfg.FUSION, normalize=self.cfg.NORMALIZE)
        self.test_set = WikiSpeciesHabitats(path=self.cfg.DATASET_PATH, split="test", merge_method=self.cfg.MERGE_METHOD, random_state=self.cfg.RANDOM_STATE, 
                                            target_base=self.cfg.TARGET_BASE, splitting=self.cfg.SPLITTING, fraction=self.cfg.FRACTION, embedding=self.cfg.EMBEDDING, 
                                            level=self.cfg.LEVEL, fusion=self.cfg.FUSION, normalize=self.cfg.NORMALIZE)
        ## Network
        self.net = MLP(size_in=self.train_set.inputs_size, size_out=self.train_set.num_classes, hidden_size=self.cfg.HIDDEN_SIZE)
        self.net.to(device=self.cfg.DEVICE)
        ## Model weights setting
        if self.cfg.LOAD_CHECKPOINT == "yes":
            name = "temp.pth"
            path = self.cfg.CHECKPOINTS_PATH + name
            self.net.load_state_dict(torch.load(path, map_location=self.cfg.DEVICE))
            print(f"Loaded model state dict from file {path}")
        ## Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.LEARNING_RATE)
        fac = 1/math.pow(10,1/3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=fac, patience=3, cooldown=0, threshold=1e-2, min_lr=self.cfg.MIN_LEARNING_RATE)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.early_stopper = EarlyStopper(patience=10, minimize=True)
        ## Loss (NB : CrossEntropyLoss is softmax AND cross entropy)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.train_set.positive_weight.to(device=self.cfg.DEVICE))
    
    def set_metrics_device(self, device):
        """Declare metrics to be used and device"""
        self.micro_F1 = MultilabelF1Score(num_labels=self.train_set.num_classes, average="micro").to(device=device) ## "overall"
        self.macro_F1 = MultilabelF1Score(num_labels=self.train_set.num_classes, average="macro").to(device=device) ## "average"
        self.classwise_F1 = MultilabelF1Score(num_labels=self.train_set.num_classes, average=None).to(device=device)
        self.classwise_P = MultilabelPrecision(num_labels=self.train_set.num_classes, average=None).to(device=device)
        self.classwise_R = MultilabelRecall(num_labels=self.train_set.num_classes, average=None).to(device=device)
        self.Hamming = MultilabelHammingDistance(num_labels=self.train_set.num_classes, average=None).to(device=device)

    def save_net_params(self, epoch):
        """Model parameters checkpoint"""
        name = "temp.pth"
        torch.save(self.net.state_dict(), self.cfg.CHECKPOINTS_PATH + name)
        
    def infer_round(self, dataloader, classwise_cf = False, global_stats = False):
        """Forward pass and metrics computation"""
        ## Set model to eval (no dropout, batchnorm fits number of samples, etc..)
        self.net.eval()
        ## Initialize metrics
        acc_loss = 0
        step_count = 0
        if global_stats:
            all_preds = torch.empty(0,self.train_set.num_classes)
            all_targets = torch.empty(0,self.train_set.num_classes)
        else:
            micro_F1 = 0
            macro_F1 = 0
            classwise_F1 = torch.zeros(self.train_set.num_classes)
            classwise_P = torch.zeros(self.train_set.num_classes)
            classwise_R = torch.zeros(self.train_set.num_classes)
            Hamming = 0
        if classwise_cf:
            cwcf = ClassWiseConfusionMatrix(num_classes=self.train_set.num_classes)
        for minibatch in dataloader:
            ## Predict batch
            input = minibatch[0].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            target = minibatch[1].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            pred = self.net(input)
            ## Update metrics
            loss = self.criterion(pred, target)
            acc_loss += loss.item()
            if global_stats:
                all_preds = torch.cat([all_preds, pred.cpu().detach()])
                all_targets = torch.cat([all_targets, target.cpu().detach()])
            else:
                micro_F1 += self.micro_F1(pred, target).cpu().detach()
                macro_F1 += self.macro_F1(pred, target).cpu().detach()
                classwise_F1 += self.classwise_F1(pred, target).cpu().detach()
                classwise_P += self.classwise_P(pred, target).cpu().detach()
                classwise_R += self.classwise_R(pred, target).cpu().detach()
                Hamming += self.Hamming(pred, target).cpu().detach()
            if classwise_cf:
                cwcf.step(target.cpu().detach(), torch.sigmoid(pred).round().cpu().detach())
            step_count += 1
        ## End of loop, return logs
        acc_loss /= step_count
        if global_stats:
            micro_F1 = self.micro_F1(all_preds, all_targets)
            macro_F1 = self.macro_F1(all_preds, all_targets)
            classwise_F1 = self.classwise_F1(all_preds, all_targets)
            classwise_P = self.classwise_P(all_preds, all_targets)
            classwise_R = self.classwise_R(all_preds, all_targets)
            Hamming = self.Hamming(all_preds, all_targets)
        else:
            micro_F1 /= step_count
            macro_F1 /= step_count
            classwise_F1 /= step_count
            classwise_P /= step_count
            classwise_R /= step_count
            Hamming /= step_count
        if classwise_cf:
            return acc_loss, micro_F1, macro_F1, classwise_F1, classwise_P, classwise_R, Hamming.sum(), cwcf.confusion_matrices
        else:
            return acc_loss, micro_F1, macro_F1, classwise_F1, classwise_P, classwise_R, Hamming.sum()
    
    def train_round(self, dataloader):
        """Forward pass, backard pass, metrics computation and parameters update"""
        ## Set model to train mode
        self.net.train()
        ## Initialize metrics
        acc_loss = 0
        micro_F1 = 0
        macro_F1 = 0
        step_count = 0
        for minibatch in dataloader:
            ## Reset gradients
            self.optimizer.zero_grad()
            ## Predict batch
            input = minibatch[0].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            target = minibatch[1].to(dtype=torch.float32, device=self.cfg.DEVICE, non_blocking=True)
            pred = self.net(input)
            ## Backpropagate loss and update parameters
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            ## Metrics update
            acc_loss += loss.item()
            micro_F1 += self.micro_F1(pred, target).cpu().detach()
            macro_F1 += self.macro_F1(pred, target).cpu().detach()
            step_count += 1
        return acc_loss/step_count, micro_F1/step_count, macro_F1/step_count

    def train(self):
        """Training loop"""
        ## Prepare for training
        train_loader = DataLoader(self.train_set, shuffle=True, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(self.val_set, shuffle=True, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
        start_time = time()
        bestmodel_score = 0
        ## Setting performance visualisation
        if self.cfg.WANDB_LOGS == "yes":
            experiment = wandb.init(project=f"{self.cfg.RUN_MODE}_{self.cfg.WANDB_PROJECT}", entity=self.cfg.WANDB_ENTITY, settings=wandb.Settings(start_method="fork"))
            experiment.config.update(self.cfg)
        pbar = trange(self.cfg.EPOCH_NUM)
        self.set_metrics_device(self.cfg.DEVICE)
        for epoch in pbar:
            ## Train
            train_loss, train_micro_F1, train_macro_F1 = self.train_round(train_loader)
            ## Validate
            val_loss, val_micro_F1, val_macro_F1, val_classwise_F1, val_classwise_P, val_classwise_R, _ = self.infer_round(val_loader, global_stats=False)
            ## Log progess
            pbar.set_description(f"train/val metrics at epoch {epoch+1} | loss : {train_loss:.2f}/{val_loss:.2f} | micro_F1 : {train_micro_F1:.2f}/{val_micro_F1:.2f}")
            ## Update learning rate
            self.scheduler.step(val_loss)
            ## Save model state
            if self.cfg.SAVE_CHECKPOINTS == "yes":
                self.save_net_params(epoch)
                """if bestmodel_score < val_macro_F1:
                    self.save_net_params(epoch)
                    bestmodel_score = val_macro_F1"""
            ## Check whether early stopping has to be performed
            if self.early_stopper.step(val_loss):
                ## Transfer logs to wandb
                table = [
                    val_classwise_F1.tolist(),
                    val_classwise_P.tolist(),
                    val_classwise_R.tolist()
                ]
                if self.cfg.WANDB_LOGS == "yes":
                    ## Create logs dict
                    logs_dict = {
                        "epoch": epoch+1,
                        "time": time()-start_time,
                        "learning rate": self.optimizer.param_groups[0]['lr'],
                        "train loss": train_loss,
                        "train micro F1": train_micro_F1,
                        "train macro F1": train_macro_F1,
                        "val loss": val_loss,
                        "val micro F1": val_micro_F1,
                        "val macro F1": val_macro_F1,
                        "classwise metrics": wandb.Table(data=table, columns=self.train_set.classes_codes.to_list(), rows = ["F1","P","R"])
                        }
                    experiment.log(logs_dict)
                exit()

        ## Transfer logs to wandb
        table = [
            val_classwise_F1.tolist(),
            val_classwise_P.tolist(),
            val_classwise_R.tolist()
        ]
        if self.cfg.WANDB_LOGS == "yes":
            ## Create logs dict
            logs_dict = {
                "epoch": epoch+1,
                "time": time()-start_time,
                "learning rate": self.optimizer.param_groups[0]['lr'],
                "train loss": train_loss,
                "train micro F1": train_micro_F1,
                "train macro F1": train_macro_F1,
                "val loss": val_loss,
                "val micro F1": val_micro_F1,
                "val macro F1": val_macro_F1,
                "classwise metrics": wandb.Table(data=table, columns=self.train_set.classes_codes.to_list(), rows = ["F1","P","R"])
                }
            experiment.log(logs_dict)
    
    def plot_confusion_matrix(self, cf_matrix, save_image=False):
        """Creation of an image out of the confusion matrix"""
        ## Plotging
        plt.figure()
        ax = sns.heatmap(cf_matrix, annot=True, annot_kws={"size":14}, cmap='Greens', cbar=True, #fmt='.0f',
                        xticklabels=["Yes","No"], yticklabels=["Yes","No"])
        ax.set(xlabel="Predicted", ylabel="True")
        ax.xaxis.tick_top()
        plt.rc('axes', labelsize=14)
        plt.tight_layout()
        if save_image==True:
            plt.savefig("confusion_matrix.png")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        image = PIL.Image.open(buffer)
        plt.close() 
        return image
    
    def test(self):
        """Testing loop"""
        ## Prepare for testing
        test_loader = DataLoader(self.test_set, shuffle=True, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
        ## Setting performance visualisation
        if self.cfg.WANDB_LOGS == "yes":
            experiment = wandb.init(project=f"{self.cfg.RUN_MODE}_{self.cfg.WANDB_PROJECT}", entity=self.cfg.WANDB_ENTITY, settings=wandb.Settings(start_method="fork"))
            experiment.config.update(self.cfg)
        ## Test
        self.set_metrics_device("cpu") ## To compute metrics on accumulated results
        test_loss, test_micro_F1, test_macro_F1, test_classwise_F1, test_classwise_P, test_classwise_R, test_Hamming, cf_matrices = self.infer_round(test_loader, classwise_cf=True, global_stats=True)
        ## Transfer logs to wandb
        table = [
            test_classwise_F1.tolist(),
            test_classwise_P.tolist(),
            test_classwise_R.tolist()
        ]
        if self.cfg.WANDB_LOGS == "yes":
            ## Create logs dict
            logs_dict = {
                "test loss": test_loss,
                "test micro F1": test_micro_F1,
                "test macro F1": test_macro_F1,
                "test macro H": test_Hamming,
                "classwise metrics": wandb.Table(data=table, columns=self.train_set.classes_codes.to_list(), rows = ["F1","P","R"])
                }
            for i in range(self.train_set.num_classes):
                logs_dict[f"cf_mat_{self.train_set.classes_codes[i]}"] = wandb.Image(self.plot_confusion_matrix(cf_matrices[i]))
            experiment.log(logs_dict)
        else:
            logs_dict = {
                "test loss": test_loss,
                "test micro F1": test_micro_F1,
                "test macro F1": test_macro_F1,
                "test macro H": test_Hamming,}
            print(logs_dict)

            

