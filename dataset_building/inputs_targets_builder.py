import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import random
from itertools import compress
tqdm.pandas()
import json
import tempfile
from species_splits_maker import SpeciesSplitsMaker


class InputsTargetsBuilder():
    """This class contains the processing steps to make inputs and targets using matched species and records"""
    def __init__(self, processed_data_path, final_data_path, level="class", filter=True, random_state=42):
        self.processed_data_path = processed_data_path
        self.final_data_path = final_data_path
        self.level = level # "class", "group" or "type"
        self.filter = filter
        self.random_state = random_state

    def load_and_merge(self):
        """Load and merge species occurences and habitat information"""
        min_values = {"class":0,"group":10,"type":100}
        ## Load data
        species_habitats_records = pd.read_json(self.processed_data_path + "species_habitats_records.json", orient="records")
        habitats_data = pd.read_json(self.processed_data_path + "habitats_data.json", orient="records").set_index("TypoCH_NUM")
        ## Merge sources
        species_habitats_records = species_habitats_records.join(habitats_data[["Class","Group_","Type","Hybrid"]], on="TypoCH_NUM", how="left")
        species_habitats_records = species_habitats_records.rename(columns={"Class": "class", "Group_":"group", "Type": "type", "Hybrid":"hybrid"})
        ## Ensure that the record is registered at the wanted level of precision
        species_habitats_records = species_habitats_records[species_habitats_records[self.level]>=min_values[self.level]]
        ## Get unique values of classes
        unique_classes = species_habitats_records[self.level].unique()
        ## Sort "alphabetically"
        unique_classes = list(map(str, unique_classes))
        unique_classes.sort()
        unique_classes = list(map(int, unique_classes))
        return species_habitats_records, habitats_data, unique_classes

    def split(self, list_a, chunk_size):
        """Split list in chunks, from https://www.programiz.com/python-programming/examples/list-chunks"""
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]

    def make_inputs(self, records):
        """Groups records by zone and retrieves unique species keys for each zone"""
        ## Grouping among grid cells and class
        records = records[["zone_id","species_key","split","shape_area",self.level]].rename(columns={self.level:"maps_based_class"})
        records_unbalanced = records.groupby(by=["zone_id","maps_based_class","split","shape_area"])["species_key"].agg("unique").reset_index()
        records_unbalanced["species_count"] = records_unbalanced["species_key"].apply(lambda x : len(x))
        ## Remove zones with too much observed species (for remaining suspicious points after the removal of rounded coordinates when processing species records)
        records_unbalanced = records_unbalanced[records_unbalanced["species_count"]<=100].reset_index().drop("index", axis=1)
        ## Divide large samples into smaller ones
        allowedSize = 10
        records = pd.DataFrame()
        for i in trange(len(records_unbalanced)):
            entry = records_unbalanced.loc[i]
            ## If lenght is ok, then just keep the sample
            if len(entry["species_key"])<=allowedSize:
                #entry["species_key"] = list(entry["species_key"])
                records = pd.concat([records, pd.DataFrame(entry).T])
            ## Otherwise, shuffle species keys and make chunks of wanted size
            else:
                newline = entry.copy()
                keys = list(entry["species_key"])
                random.Random(self.random_state).shuffle(keys)
                for chunk in list(self.split(keys,chunk_size=allowedSize)):
                    newline["species_key"] = chunk
                    records = pd.concat([records, pd.DataFrame(newline).T])

        records["species_count"] = records["species_key"].apply(lambda x : len(x))
        records = records.reset_index().drop("index", axis=1)
        print(records.head())
        return records

    def get_onehots(self, entry, unique_classes):
        """Retrieve one_hot encodings"""
        x = [unique_classes.index(c) for c in entry]
        one_hots = F.one_hot(torch.tensor(x), num_classes=len(unique_classes)).sum(dim=0).tolist()
        return one_hots

    def get_species_classes(self, records, unique_classes):
        """Get one_hot encoded classes for each species"""
        if self.filter:
            species_classes_counts = pd.DataFrame(records[["species_key",self.level]].value_counts()).reset_index().rename(columns={0:"count"})
            species_classes_counts = species_classes_counts.join(pd.DataFrame(records["species_key"].value_counts()).rename(columns={"species_key":"total"})["total"], on="species_key", how="inner")
            species_classes_counts["fraction"] = species_classes_counts["count"]/species_classes_counts["total"]
            records = species_classes_counts[species_classes_counts["fraction"]>0.01]
        species_classes = records.groupby("species_key")[self.level].unique().reset_index().rename(columns={self.level:"classes"})
        species_classes["classes_onehot"] = species_classes["classes"].apply(lambda x : self.get_onehots(x, unique_classes))
        return species_classes
    
    def intersect_species_classes(self, species_list, species, num_classes):
        """Build targets using one hot encoded classes"""
        target = torch.ones(num_classes)
        for key in species_list:
            mask = torch.tensor(species[species["species_key"]==key]["classes_onehot"].to_list()[0])
            target = target*mask
        return target.int().tolist()
    
    def get_species_habitats_ids(self, inputs, unique_classes, habitats_data):
        """Assign new ID to all present species"""
        species_ids = []
        for i in trange(len(inputs)):
            species_ids = species_ids+ list(inputs["species_key"].loc[i])
        species_ids = pd.DataFrame(species_ids)
        species_ids = species_ids.drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index":"ID", 0:"species_key"})
        print(species_ids.head())
        habitats_ids = pd.DataFrame(unique_classes).rename(columns={0:"class"})
        habitats_ids = habitats_ids.join(habitats_data[["TypoCH_DE","TypoCH_FR","TypoCH_IT"]], on="class", how="left")
        print(habitats_ids.head())
        return species_ids, habitats_ids

    def make_species_based_targets(self, species_classes, inputs, unique_classes):
        """Make targets using intersection of habitats where species occur for each zone"""
        print(species_classes.head())
        inputs["species_based_class"] = inputs["species_key"].progress_apply(lambda x : self.intersect_species_classes(x,species_classes,len(unique_classes)))
        inputs["species_based_num_classes"] = inputs["species_based_class"].progress_apply(lambda x : int(torch.tensor(x).sum()))
        print(inputs.head())
        return inputs

    def make_set_based_targets(self, pairs, unique_classes):
        """Make targets using union of habitats where each set of species occurred"""
        num_classes = len(unique_classes)
        ## List to string for grouping
        pairs["species_key"] = pairs["species_key"].apply(lambda x : json.dumps(x))
        ## Get unique habitats per set
        temp = pd.DataFrame(pairs.groupby("species_key")["maps_based_class"].unique()).reset_index()
        print(temp.head())
        temp["set_based_num_classes"] = temp["maps_based_class"].apply(lambda x : len(x))
        ## One_hot encoding
        temp["set_based_class"] = temp["maps_based_class"].apply(lambda x : self.get_onehots(x,unique_classes=unique_classes))
        #temp.to_json(f"processed_data/temp_{self.level}.json", orient="records")
        ## Joint new targets
        pairs = pairs.join(temp.set_index("species_key")[["set_based_class","set_based_num_classes"]], on="species_key", how="inner")## List to string for grouping
        pairs["species_key"] = pairs["species_key"].apply(lambda x : json.loads(x))
        return pairs
    
    def remove_unwanted_classes(self, record, mask):
        return list(compress(record,mask))

    def process_data(self):
        """Process records"""
        ## Load data
        records, habitats_data, unique_classes = self.load_and_merge()
        print("Loaded data")
        ## Make inputs (groupby polygon with unique species as aggregate)
        inputs = self.make_inputs(records)
        print("Made inputs with max 10 species")
        ## Make species based targets
        ## Get classes for species
        species_classes = self.get_species_classes(records, unique_classes)
        inputs_targets = self.make_species_based_targets(species_classes, inputs, unique_classes)
        ## Get species and habitats keys and names (for the ones actually present in the dataset)
        species_ids, habitats_ids = self.get_species_habitats_ids(inputs_targets, unique_classes, habitats_data)
        ## To fix lists, arrays and dtypes issues in species_keys
        with tempfile.TemporaryDirectory() as temp_dir_path:
            inputs_targets.to_json(temp_dir_path+"/temp.json", orient="records")
            inputs_targets = pd.read_json(temp_dir_path+"/temp.json", orient="records")
        spatial_inputs_targets = self.make_set_based_targets(inputs_targets,unique_classes)
        print(spatial_inputs_targets.head())
        print(habitats_ids.head())
        ## Split data again using species
        ssm = SpeciesSplitsMaker(inputs_targets=spatial_inputs_targets.copy(), species_keys=species_ids, random_state=self.random_state)
        species_inputs_targets = ssm.process()
        ## Re compute targets to avoid conflicting examples
        species_inputs_targets = self.make_species_based_targets(species_classes, species_inputs_targets.drop(['set_based_class', 'set_based_num_classes'], axis=1), unique_classes)
        species_inputs_targets = self.make_set_based_targets(species_inputs_targets, unique_classes)

        """train = self.make_species_based_targets(species_classes, train.drop(['set_based_class', 'set_based_num_classes'], axis=1), unique_classes)
        train = self.make_set_based_targets(train,unique_classes)
        val = self.make_species_based_targets(species_classes, val.drop(['set_based_class', 'set_based_num_classes'], axis=1), unique_classes)
        val = self.make_set_based_targets(val,unique_classes)
        test = self.make_species_based_targets(species_classes, test.drop(['set_based_class', 'set_based_num_classes'], axis=1), unique_classes)
        test = self.make_set_based_targets(test,unique_classes)

        train["split"] = "train"
        val["split"] = "val"
        test["split"] = "test"
        species_inputs_targets = pd.concat([train, val, test])"""
        return spatial_inputs_targets, species_inputs_targets, species_ids, habitats_ids

