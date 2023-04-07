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


class HabitatMapsProcessor():
    """This class contains the processing steps of the cantonal Habitat maps"""
    def __init__(self, study_area, cantons_list, raw_data_path):
        self.study_area = study_area
        self.cantons_list = cantons_list
        self.raw_data_path = raw_data_path
        self.grid = GridBuilder().grid_from_shape(shape=self.study_area, split=False)
    
    def clip(self, shape):
        """Clip fast using dask-geopandas"""
        return dgpd.clip(gdf=dgpd.from_geopandas(shape, npartitions=8), mask=self.study_area).compute()
    
    def groupby_gridcell(self, shape):
        """ Group all polygons of same type among each cell"""
        ## Intersecting with grid cells
        intersected = dgpd.from_geopandas(shape, npartitions=8).sjoin(dgpd.from_geopandas(self.grid, npartitions=8), how="inner").compute()
        shape = None
        intersected = intersected[["TypoCH_NUM", "index_right", "canton", "Shape_Area", "geometry"]]
        intersected = intersected.rename(columns={"index_right": "grid_id"})
        ## Merge shapes of same type within cells (with for loop to alleviate memory usage)
        merged = pd.DataFrame()
        for i in tqdm(intersected["grid_id"].unique()):
            subset = intersected[intersected["grid_id"]==i]
            subset = dgpd.from_geopandas(subset, npartitions=8).dissolve(by=["TypoCH_NUM", "grid_id", "canton"], aggfunc={"Shape_Area": "sum"}).compute()
            merged = pd.concat([merged, subset])
            intersected = intersected.drop(intersected[intersected["grid_id"]==i].index)
        #merged = dgpd.from_geopandas(intersected, npartitions=8).dissolve(by=["TypoCH_NUM", "grid_id", "canton"], aggfunc={"Shape_Area": "sum"})
        #merged = merged.compute()
        return merged

    def process_cantons(self):
        """Processing of cantonal maps"""
        habitats_map = pd.DataFrame()
        habitats_data = pd.DataFrame()
        for canton in self.cantons_list:
            print(f"Processing canton {canton}")
            ## Habitat geometries
            habitats = gpd.read_file(self.raw_data_path+f"habitatmap_{canton.lower()}/HabitatMap_{canton}.gdb/", layer=0)
            habitats["canton"] = canton
            habitats = self.clip(habitats)
            print("Clipped maps")
            ## Grouping by cell is currently not enabled
            habitats = self.groupby_gridcell(habitats)
            #habitats = habitats[["TypoCH_NUM", "canton", "Shape_Area", "geometry"]]
            print("Grouped maps by grid cell")
            habitats_map = pd.concat([habitats_map, habitats])
            ## Habitat descriptions
            unique_habitats = gpd.read_file(self.raw_data_path+f"habitatmap_{canton.lower()}/HabitatMap_{canton}.gdb/", layer=1)
            habitats_data = pd.DataFrame(pd.concat([habitats_data, unique_habitats]).drop_duplicates())
            print("Got unique habitats")
        habitats_map = habitats_map.reset_index().reset_index().rename(columns={"index":"zone_id"})
        habitats_data = habitats_data.drop("geometry", axis=1)
        return habitats_map, habitats_data


class SpeciesRecordsProcessor():
    """This class contains the processing steps of the georeferenced species observations"""
    def __init__(self, study_area, raw_data_path, final_data_path):
        self.study_area = study_area
        self.raw_data_path = raw_data_path
        self.final_data_path = final_data_path
        self.grid = GridBuilder().grid_from_shape(shape=self.study_area, split=True)
    
    def filter(self, records):
        """Filter species records"""
        ## Renaming fields
        records = records.rename(columns={"speciesKey":"species_key","gbifID":"gbif_id","scientificName":"scientific_name",
                                          "decimalLatitude":"lat","decimalLongitude":"long","eventDate":"event_date"})
        # Filtering on date
        records = records[records["year"]>=1950]
        # Filtering on taxonomy level
        records = records.dropna(subset=["species","species_records"])
        records["species_records"] = records["species_records"].apply(lambda x : int(x))
        # Keeping meaningful fields
        fields=["gbif_id", "scientific_name", "kingdom", "phylum", "class", "order", "family", 
                "genus", "species", "lat", "lon", "event_date", "species_records"]
        records = records[fields]
        return records

    def reproject(self, records, old="EPSG:4326", new="EPSG:2056"):
        """Reprojecting shape from old to new CRS"""
        ## Set transformation
        trans = Transformer.from_crs(old, new, always_xy=True)
        ## Reproject
        xx, yy = trans.transform(records["long"].values, records["lat"].values)
        records["E"] = xx
        records["N"] = yy
        ## Wrap in GeoDataFrame
        records = gpd.GeoDataFrame(records, geometry=gpd.points_from_xy(records.E, records.N))
        records = records.set_crs(crs=new)
        return records
    
    def clip(self, shape):
        """Clip fast using dask-geopandas"""
        return dgpd.clip(gdf=dgpd.from_geopandas(shape, npartitions=8), mask=self.study_area).compute()
    
    def trim_species(self, records):
        """Keep only species that have wikipedia info"""
        ## List all documented species
        species_list = [int(elem[:-5]) for elem in os.listdir(self.final_data_path+"species/")]
        ## Keep entries where species key are known
        records = records[records["species_records"].isin(species_list)]
        return records
    
    def add_grid(self, shape):
        ## Intersecting with grid cells
        intersected = dgpd.from_geopandas(shape, npartitions=8).sjoin(dgpd.from_geopandas(self.grid, npartitions=8), how="inner").compute()
        intersected = intersected.rename(columns={"index_right": "grid_id"})
        return intersected
    
    def process_records(self):
        """Processing of species records"""
        ## Loading data
        species_records = pd.read_csv(self.raw_data_path+f"gbif_raw.csv", sep="\t")
        ## Filtering
        species_records = self.filter(species_records)
        print("Filtered records")
        ## Reproject
        species_records = self.reproject(species_records)
        print("Reprojected records")
        ## Trim observed species
        species_records = self.trim_species(species_records)
        print("Filtered records wrt species")
        ## Clip using study area
        species_records = self.clip(species_records)
        print("Clipped records")
        ## Add grid cell id to records
        species_records = self.add_grid(species_records)
        print("Added grid ID to records")
        return species_records


class SpeciesHabitatMerger():
    """This class is intended to merge processed species and natural habitats data"""
    def __init__(self):
        pass

    def merge_data(self, species, habitats):
        ## Performing inner spatial join
        #species_habitats_records = dgpd.from_geopandas(species, npartitions=8).sjoin(dgpd.from_geopandas(habitats, npartitions=8), how="inner").compute()
        ## Performing inner spatial join (with for loop to alleviate memory usage)
        species_habitats_records = pd.DataFrame()
        for i in tqdm(species["grid_id"].unique()):
            subset_species = species[species["grid_id"]==i]
            subset_habitats = habitats[habitats["grid_id"]==i]
            subset_species = subset_species.sjoin(subset_habitats, how="inner")
            ## Multiprocessing is useless with small batches like this
            #subset_species = dgpd.from_geopandas(subset_species, npartitions=8).sjoin(dgpd.from_geopandas(subset_habitats, npartitions=8), how="inner").compute()
            species_habitats_records = pd.concat([species_habitats_records, subset_species])
            species = species.drop(species[species["grid_id"]==i].index)
            habitats = habitats.drop(habitats[habitats["grid_id"]==i].index)
        return species_habitats_records
    

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
        species_habitats_records = species_habitats_records[["zone_id", "grid_id_right", "TypoCH_NUM", "species_key", "Shape_Area", "canton"]]
        ## Renaming columns
        species_habitats_records = species_habitats_records.rename(columns={"grid_id_right":"grid_id", "Shape_Area": "shape_area"})
        print("Trimmed fields")
        ## Saving species and habitats pairs
        species_habitats_records.to_json(processed_data_path+"species_habitats_records.json", orient="records")
        print("Saved species_habitats_records.json, head is the following")
        print(species_habitats_records.head())
        """
        ## Saving species recorded in each zone
        species_habitats_records.groupby(["zone_id","TypoCH_NUM"])["species_records"].agg(["unique"]).reset_index().to_json(final_data_path+"speciesInZones.json", orient="records")
        print("Saved speciesInZones.json, head is the following")
        print(species_habitats_records.groupby(["zone_id","TypoCH_NUM"])["species_records"].agg(["unique"]).reset_index().head())
        ## Saving species recorded in each habitat type
        species_habitats_records.groupby(["TypoCH_NUM"])["species_records"].agg(["unique"]).reset_index().to_json(processed_data_path+"speciesInHabitats.json", orient="records")
        print("Saved speciesInHabitats.json, head is the following")
        print(species_habitats_records.groupby(["TypoCH_NUM"])["species_records"].agg(["unique"]).reset_index())
        ## Saving habitats recorded for each species
        species_habitats_records.groupby(["species_records"])["TypoCH_NUM"].agg(["unique"]).reset_index().to_json(processed_data_path+"habitatsOfSpecies.json", orient="records")
        print("Saved habitatsOfSpecies.json, head is the following")
        print(species_habitats_records.groupby(["species_records"])["TypoCH_NUM"].agg(["unique"]).reset_index())
        """

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