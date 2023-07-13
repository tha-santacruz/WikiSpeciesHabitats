from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dataset import BertTuningDataset
from torch.utils.data import DataLoader
import torch


## Dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
preprocessing = lambda x : tokenizer(x, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
val_set = BertTuningDataset(path="/data/nicola/WSH/bert_finetuning/", split="val", preprocessing=preprocessing)
dataloader = DataLoader(val_set, batch_size=1, shuffle=False)

## Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device="cuda:0")
model.load_state_dict(torch.load("/data/nicola/WSH/checkpoints/bestmodel_distilbert.pth"))
model.eval()

categories = ["Negative","Positive"]

for minibatch in dataloader:
    ## Predict batch
    inp = minibatch[0].to(dtype=torch.int64, device="cuda:0", non_blocking=True)
    mask = minibatch[1].to(dtype=torch.int64, device="cuda:0", non_blocking=True)
    target = minibatch[2].to(dtype=torch.float32, device="cpu", non_blocking=True)
    text = minibatch[3]
    pred = torch.nn.functional.sigmoid(model(input_ids = inp, attention_mask = mask).logits).cpu().detach()

    #print("\n")
    #print(text)
    if categories[target.argmax(dim=1).squeeze(dim=0)] != categories[pred.argmax(dim=1).squeeze(dim=0)]:
        print("WRONG - ")
    print(text)
    print(f"true : {categories[target.argmax(dim=1).squeeze(dim=0)]}, predicted : {categories[pred.argmax(dim=1).squeeze(dim=0)]}, score {pred.max():.3f}")
    #input("continue by pressing key")

#phrase = preprocessing("I went hiking in the forest last week")
#phrase = preprocessing("This species lives in forests but also gathers in groups in open fields")
#score = torch.nn.functional.sigmoid(model(phrase.input_ids.to(dtype=torch.int64, device="cuda:0"), phrase.attention_mask.to(dtype=torch.int64, device="cuda:0")).logits)

#print(float(score.squeeze().cpu().detach()[1]))

