## Standard Library
import argparse
import json
import os
import sys

## External packages
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm, trange

## Custom modules
from grid import GridBuilder
from habitat_maps_processor import HabitatMapsProcessor
from inputs_targets_builder import InputsTargetsBuilder
from species_habitat_merger import SpeciesHabitatMerger
from species_records_processor import SpeciesRecordsProcessor


class Step1():
    """Step 1 : Habitat maps processing"""
    def __init__(self):
        print("Step 1 : Habitat maps processing")
        ## Paths
        raw_data_path = "./raw_data/"
        processed_data_path = "./processed_data/"
        final_data_path = "./final_data/"
        ## Area of interest
        study_area = gpd.read_file("./raw_data/study_area/study_area.shp")
        ## Habitat maps processing
        hmp = HabitatMapsProcessor(study_area=study_area, cantons_list=["VS","VD"], raw_data_path=raw_data_path)
        habitats_map, habitats_data = hmp.process_cantons()
        hmp = None
        ## Saving data
        habitats_map.to_file(processed_data_path+"habitats_map.gpkg", driver="GPKG")
        print("Saved habitats_map.gpkg, head is the following")
        print(habitats_map.head())
        habitats_data.to_json(final_data_path+"habitats_data.json", orient="records")
        print("Saved habitats_data.json, head is the following")
        print(habitats_data.head())


class Step2():
    """Step 2 : Species records processing"""
    def __init__(self):
        print("Step 2 : Species records processing")
        ## Paths
        raw_data_path = "./raw_data/"
        processed_data_path = "./processed_data/"
        final_data_path = "./final_data/"
        ## Area of interest
        study_area = gpd.read_file("./raw_data/study_area/study_area.shp")
        ## Species records processing
        srp = SpeciesRecordsProcessor(study_area=study_area, raw_data_path=raw_data_path, final_data_path=final_data_path)
        species_records = srp.process_records()
        srp = None
        ## Saving data and unique species records
        species_records.to_file(processed_data_path+"species_records.gpkg", driver="GPKG")
        print("Saved species_records.gpkg, head is the following")
        print(species_records.head())
        fields = ["scientific_name","kingdom","phylum","class","order","family","genus","species","species_key"]
        species_data = species_records[fields].drop_duplicates()
        species_data.to_json(final_data_path+"species_data.json")
        print("Saved species_data.json, head is the following")
        print(species_data.head())


class Step3():
    """Step 3 : Intersecting species and records"""
    def __init__(self):
        print("Step 3 : Intersecting species and records")
        ## Paths
        processed_data_path = "./processed_data/"
        final_data_path = "./final_data/"
        ## Species-habitats merger
        shm = SpeciesHabitatMerger()
        ## Grid
        study_area = gpd.read_file("./raw_data/study_area/study_area.shp")
        grid = GridBuilder().grid_from_shape(shape=study_area, width=25000, height=25000)
        ## Process by big chunks to alleviate memory usage
        species_habitats_records = pd.DataFrame()
        for i in trange(len(grid)):
            gridCell = grid.loc[i].geometry
            ## Loading data
            habitats_map = gpd.read_file(processed_data_path+"habitats_map.gpkg", mask=gridCell)
            species_records = gpd.read_file(processed_data_path+"species_records.gpkg", mask=gridCell)
            print("Loaded records")
            ## Species-habitats pairs
            newRecords = shm.merge_data(species=species_records, habitats=habitats_map)
            species_habitats_records = pd.concat([species_habitats_records, newRecords])
        print("Intersected data")
        ## Keeping meaningful columns only
        species_habitats_records = species_habitats_records[["zone_id", "grid_id_right", "TypoCH_NUM", "species_key", "Shape_Area", "canton", "split"]]
        ## Renaming columns
        species_habitats_records = species_habitats_records.rename(columns={"grid_id_right":"grid_id", "Shape_Area": "shape_area"})
        print("Trimmed fields")
        ## Saving species and habitats pairs
        species_habitats_records.to_json(processed_data_path+"species_habitats_records.json", orient="records")
        print("Saved species_habitats_records.json, head is the following")
        print(species_habitats_records.head())
        """
        ## Saving species recorded in each zone
        species_habitats_records.groupby(["zone_id","TypoCH_NUM"])["species_key"].agg(["unique"]).reset_index().to_json(final_data_path+"speciesInZones.json", orient="records")
        print("Saved speciesInZones.json, head is the following")
        print(species_habitats_records.groupby(["zone_id","TypoCH_NUM"])["species_key"].agg(["unique"]).reset_index().head())
        ## Saving species recorded in each habitat type
        species_habitats_records.groupby(["TypoCH_NUM"])["species_key"].agg(["unique"]).reset_index().to_json(processed_data_path+"speciesInHabitats.json", orient="records")
        print("Saved speciesInHabitats.json, head is the following")
        print(species_habitats_records.groupby(["TypoCH_NUM"])["species_key"].agg(["unique"]).reset_index())
        ## Saving habitats recorded for each species
        species_habitats_records.groupby(["species_key"])["TypoCH_NUM"].agg(["unique"]).reset_index().to_json(processed_data_path+"habitatsOfSpecies.json", orient="records")
        print("Saved habitatsOfSpecies.json, head is the following")
        print(species_habitats_records.groupby(["species_key"])["TypoCH_NUM"].agg(["unique"]).reset_index())
        """


class Step4():
    """Step 4 : Creating inputs and targets for splits"""   
    def __init__(self):
        print("Step 4 : Making split files")
        ## Paths
        processed_data_path = "./processed_data/"
        final_data_path = "./final_data/"
        ## Examples builder
        itb = InputsTargetsBuilder()


if __name__=="__main__":
    """
    Global Biodiversity Information Facility x The Habitat Map of Switzerland x Wikipedia
    Creation of a dataset containing textual representation of species present in habitat zones over a study area
    """
    ## Parser for the execution of the steps
    parser = argparse.ArgumentParser(description="Dataset building parser")
    parser.add_argument("--STEP", dest="STEP",
                        choices=["1","2","3","all"],
                        help="{STEP 1 : habitat maps processing, SETP 2 : species maps processing, STEP 3 : habitat and species STEP all : all steps}",
                        type=str)
    args = parser.parse_args()

    print("Creating dataset...")
    
    print(f"Performing step : {args.STEP}")
    if args.STEP == "1":
        Step1()
    elif args.STEP == "2":
        Step2()
    elif args.STEP == "3":
        Step3()
    elif args.STEP == "all":
        Step1()
        Step2()
        Step3()

    print("Dataset is finished")