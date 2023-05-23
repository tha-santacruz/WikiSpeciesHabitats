import pandas as pd
import torch
import json

class SpeciesSplitsMaker():
    """Split data again using species instead of spatial information"""
    def __init__(self, inputs_targets, species_keys, random_state=42):
        ## Get half of species
        a = species_keys.set_index("ID")
        self.random_state = random_state
        self.half1 = a.sample(frac=0.5, random_state=self.random_state)
        self.half2 = a.drop(self.half1.index)
        self.all_data = inputs_targets

    def retrieve_in_ref_list(self, species_list, ref_list):
        out = [spe for spe in species_list if spe in ref_list]
        return out
    
    def process(self):
        ## Split species keys in two sets e.g. [S1, S2, S3, S4, S5] -> [S2, S3] [S1, S4, S5]
        self.all_data["half1_species"] = self.all_data["species_key"].apply(lambda x : self.retrieve_in_ref_list(x,self.half1["species_key"].to_list()))
        self.all_data["half2_species"] = self.all_data["species_key"].apply(lambda x : self.retrieve_in_ref_list(x, self.half2["species_key"].to_list()))
        ## Retrieving new number of species per sample
        self.all_data["len_half1"] = self.all_data["half1_species"].apply(lambda x : len(x))
        self.all_data["len_half2"] = self.all_data["half2_species"].apply(lambda x : len(x))
        ## Split examples of two halves
        split_1 = self.all_data.loc[self.all_data["len_half1"]>0].drop(["len_half2","half2_species","species_key","species_count"], axis=1).rename(columns={"half1_species":"species_key","len_half1":"species_count"})
        split_2 = self.all_data.loc[self.all_data["len_half2"]>0].drop(["len_half1","half1_species","species_key","species_count"], axis=1).rename(columns={"half2_species":"species_key","len_half2":"species_count"})
        ## Split 1 is for train and val (90%, 10%)
        val_set = split_1.sample(frac=0.1, random_state=self.random_state)
        train_set = split_1.drop(val_set.index)
        ## Remove redundant examples from the validation set
        train_set["species_key"] = train_set["species_key"].apply(lambda x : json.dumps(x))
        val_set["species_key"] = val_set["species_key"].apply(lambda x : json.dumps(x))
        train_pops = train_set["species_key"].to_list()
        val_pops = val_set["species_key"].to_list()
        redundant = [a for a in val_pops if a in train_pops]
        val_set = val_set[~val_set["species_key"].isin(redundant)]
        train_set["species_key"] = train_set["species_key"].apply(lambda x : json.loads(x))
        val_set["species_key"] = val_set["species_key"].apply(lambda x : json.loads(x))
        ## Split 2 is for testing
        test_set = split_2
        return train_set, val_set, test_set





if __name__ == "__main__":
    inp_tar = pd.read_json("/data/nicola/WSH/final_data/L1_all_data.json", orient="records")
    spe_key = pd.read_json("/data/nicola/WSH/final_data/L1_species_keys.json", orient="records")

    spm = SpeciesSplitsMaker(inputs_targets=inp_tar, species_keys=spe_key)
    a, b, c = spm.process()
    print(len(a))
    print(len(b))
    print(len(c))
    for split in [a,b,c]:
        classes = torch.tensor(split["species_based_class"].to_list())
        print(classes.sum(dim=0).div(classes.size(0)))



