from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import torch
from sklearn.metrics import confusion_matrix

class ClassWiseConfusionMatrix():
    def __init__(self, num_classes):
        """Computes TP, TN, FP, FN for each class"""
        ## Number of classes
        self.num_classes = num_classes
        ## Confusion_matrices
        self.confusion_matrices = torch.zeros(num_classes,2,2)
    
    def step(self, target, pred):
        """Update confidence matrices of all classes"""
        for i in range(self.num_classes):
            cf = confusion_matrix(target[:,i].numpy(), pred[:,i].numpy(), labels=[1,0])
            self.confusion_matrices[i] += torch.from_numpy(cf)


class EarlyStopper():
    def __init__(self, patience=5, minimize=True):
        """Stops the execution of the script when overfitting is reached"""
        ## Number of epochs tolerated while missing the objective
        self.patience = patience
        ## Objective w.r.t the monitored metric
        self.minimize = True
        ## Number of consecutive epoches where the objective has been missed
        self.counter = 0
        ## Reference lowest value of the monitored metric yet
        self.ref_value = 1e12
    
    def step(self, new_value):
        """Update counter and reference value and stop early"""
        if self.minimize:
            if new_value < self.ref_value:
                self.ref_value = new_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if new_value > self.ref_value:
                self.ref_value = new_value
                self.counter = 0
            else:
                self.counter += 1
        if self.counter == self.patience:
            return True
        else:
            return False


class WikiTextCleaner():
    """Remove templates and unwanted text from textual inputs"""
    def __init__(self, remove_stopwords=True):
        self.split_words = ["External links", "References", "See also"]
        self.forbidden_words = ["==", "\n"]
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.tokenizer = word_tokenize
            self.stop_words = stopwords.words("english")

    def clean_text(self, text, cut_useless_paragraphs=True):
        ## Cut out references
        if cut_useless_paragraphs:
            for word in self.split_words:
                text = text.split(word)[0]
        ## Remove remaining templates
        text = re.sub('<[^>]+>', '', text)
        ## Remove stopwords
        if self.remove_stopwords:
            word_tokens = self.tokenizer(text)
            filtered = [w for w in word_tokens if w.lower() not in self.stop_words]
            text = " ".join(filtered)
        ## Remove unwanted words
        for word in self.forbidden_words:
            text = text.replace(word, "")
        ## Remove multiple spaces
        text = re.sub('\ \ +', ' ', text)
        return text
    
if __name__=="__main__":
    """vals_list = [10,9,3,4,5,6,7,8,9,10,11,12,13]
    es = EarlyStopper(patience=3)
    for val in vals_list:
        print(val)
        es.step(val)"""
    out = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])
    tar = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    cf = ClassWiseConfusionMatrix(num_classes=4)
    cf.step(tar, out)
    cf.step(tar, out)
    print(cf.confusion_matrices)