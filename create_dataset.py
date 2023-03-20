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
from tqdm import tqdm

## Custom modules
from grid import GridBuilder


class HabitatMapsProcessor():
    """This class contains the processing steps of the cantonal Habitat maps"""
    def __init__(self, studyArea, cantonsList, rawDataPath):
        self.studyArea = studyArea
        self.cantonsList = cantonsList
        self.rawDataPath = rawDataPath
        self.grid = GridBuilder().grid_from_shape(shape=self.studyArea)
    
    def clip(self, shape):
        """Clip fast using dask-geopandas"""
        return dgpd.clip(gdf=dgpd.from_geopandas(shape, npartitions=8), mask=self.studyArea).compute()
    
    def groupby_gridcell(self, shape):
        """ Group all polygons of same type among each cell"""
        ## Intersecting with grid cells
        intersected = dgpd.from_geopandas(shape, npartitions=8).sjoin(dgpd.from_geopandas(self.grid, npartitions=8), how="inner").compute()
        intersected = intersected[["TypoCH_NUM", "index_right", "canton", "Shape_Area", "geometry"]]
        intersected = intersected.rename(columns={"index_right": "gridID"})
        ## Merge shapes of same type within cells (with for loop to alleviate memory usage)
        merged = pd.DataFrame()
        for i in tqdm(intersected["gridID"].unique()):
            subset = intersected[intersected["gridID"]==i]
            subset = dgpd.from_geopandas(subset, npartitions=8).dissolve(by=["TypoCH_NUM", "gridID", "canton"], aggfunc={"Shape_Area": "sum"}).compute()
            merged = pd.concat([merged, subset])
        #merged = dgpd.from_geopandas(intersected, npartitions=8).dissolve(by=["TypoCH_NUM", "gridID", "canton"], aggfunc={"Shape_Area": "sum"})
        #merged = merged.compute()
        return merged

    def process_cantons(self):
        """Processing of cantonal maps"""
        habitatsMap = pd.DataFrame()
        habitatsData = pd.DataFrame()
        for canton in self.cantonsList:
            print(f"Processing canton {canton}")
            ## Habitat geometries
            habitats = gpd.read_file(self.rawDataPath+f"habitatmap_{canton.lower()}/HabitatMap_{canton}.gdb/", layer=0)
            habitats["canton"] = canton
            habitats = self.clip(habitats)
            print("Clipped maps")
            ## Grouping by cell is currently not enabled
            habitats = self.groupby_gridcell(habitats)
            #habitats = habitats[["TypoCH_NUM", "canton", "Shape_Area", "geometry"]]
            print("Grouped maps by grid cell")
            habitatsMap = pd.concat([habitatsMap, habitats])
            ## Habitat descriptions
            uniqueHabitats = gpd.read_file(self.rawDataPath+f"habitatmap_{canton.lower()}/HabitatMap_{canton}.gdb/", layer=1)
            habitatsData = pd.DataFrame(pd.concat([habitatsData, uniqueHabitats]).drop_duplicates())
            print("Got unique habitats")
        habitatsMap = habitatsMap.reset_index()
        habitatsData = habitatsData.drop("geometry", axis=1)
        return habitatsMap, habitatsData


class SpeciesRecordsProcessor():
    """This class contains the processing steps of the georeferenced species observations"""
    def __init__(self, studyArea, rawDataPath, finalDataPath):
        self.studyArea = studyArea
        self.rawDataPath = rawDataPath
        self.finalDataPath = finalDataPath
        self.grid = GridBuilder().grid_from_shape(shape=self.studyArea)
    
    def filter(self, records):
        """Filter species records"""
        # Filtering on date
        records = records[records["year"]>=1950]
        # Filtering on taxonomy level
        records = records.dropna(subset=["species","speciesKey"])
        records["speciesKey"] = records["speciesKey"].apply(lambda x : int(x))
        # Keeping meaningful fields
        fields=["gbifID", "scientificName", "kingdom", "phylum", "class", "order", "family", 
                "genus", "species", "decimalLatitude", "decimalLongitude", "eventDate", "speciesKey"]
        records = records[fields]
        return records

    def reproject(self, records, old="EPSG:4326", new="EPSG:2056"):
        """Reprojecting shape from old to new CRS"""
        ## Set transformation
        trans = Transformer.from_crs(old, new, always_xy=True)
        ## Reproject
        xx, yy = trans.transform(records["decimalLongitude"].values, records["decimalLatitude"].values)
        records["E"] = xx
        records["N"] = yy
        ## Wrap in GeoDataFrame
        records = gpd.GeoDataFrame(records, geometry=gpd.points_from_xy(records.E, records.N))
        records = records.set_crs(crs=new)
        return records
    
    def clip(self, shape):
        """Clip fast using dask-geopandas"""
        return dgpd.clip(gdf=dgpd.from_geopandas(shape, npartitions=8), mask=self.studyArea).compute()
    
    def trim_species(self, records):
        """Keep only species that have wikipedia info"""
        ## List all documented species
        speciesList = [int(elem[:-5]) for elem in os.listdir(self.finalDataPath+"species/")]
        ## Keep entries where species key are known
        records = records[records["speciesKey"].isin(speciesList)]
        return records
    
    def add_grid(self, shape):
        ## Intersecting with grid cells
        intersected = dgpd.from_geopandas(shape, npartitions=8).sjoin(dgpd.from_geopandas(self.grid, npartitions=8), how="inner").compute()
        intersected = intersected.rename(columns={"index_right": "gridID"})
        return intersected
    
    def process_records(self):
        """Processing of species records"""
        ## Loading data
        speciesRecords = pd.read_csv(self.rawDataPath+f"gbif_raw.csv", sep="\t")
        ## Filtering
        speciesRecords = self.filter(speciesRecords)
        print("Filtered records")
        ## Reproject
        speciesRecords = self.reproject(speciesRecords)
        print("Reprojected records")
        ## Trim observed species
        speciesRecords = self.trim_species(speciesRecords)
        print("Filtered records wrt species")
        ## Clip using study area
        speciesRecords = self.clip(speciesRecords)
        print("Clipped records")
        ## Add grid cell id to records
        speciesRecords = self.add_grid(speciesRecords)
        print("Added grid ID to records")
        return speciesRecords


class SpeciesHabitatMerger():
    """This class is intended to merge processed species and natural habitats data"""
    def __init__(self):
        pass

    def merge_data(self, species, habitats):
        ## Performing inner spatial join
        #speciesHabitatsRecords = dgpd.from_geopandas(species, npartitions=8).sjoin(dgpd.from_geopandas(habitats, npartitions=8), how="inner").compute()
        ## Performing inner spatial join (with for loop to alleviate memory usage)
        speciesHabitatsRecords = pd.DataFrame()
        for i in tqdm(species["gridID"].unique()):
            subsetSpecies = species[species["gridID"]==i]
            subsetHabitats = habitats[habitats["gridID"]==i]
            subsetSpecies = subsetSpecies.sjoin(subsetHabitats, how="inner")
            ## Multiprocessing is useless with small batches like this
            #subsetSpecies = dgpd.from_geopandas(subsetSpecies, npartitions=8).sjoin(dgpd.from_geopandas(subsetHabitats, npartitions=8), how="inner").compute()
            speciesHabitatsRecords = pd.concat([speciesHabitatsRecords, subsetSpecies])
            species = species.drop(species[species["gridID"]==i].index)
            habitats = habitats.drop(habitats[habitats["gridID"]==i].index)
        print("Intersected data")
        ## Keeping meaningful columns only
        speciesHabitatsRecords = speciesHabitatsRecords[["index_right", "TypoCH_NUM", "speciesKey", "Shape_Area", "canton"]]
        ## Renaming columns
        speciesHabitatsRecords = speciesHabitatsRecords.rename(columns={"index_right": "zoneID","Shape_Area": "shapeArea"})
        print("Trimmed fields")
        return speciesHabitatsRecords
    

class Step1():
    """Step 1 : Habitat maps processing"""
    def __init__(self):
        print("Step 1 : Habitat maps processing")
        ## Paths
        rawDataPath = "./raw_data/"
        processedDataPath = "./processed_data/"
        finalDataPath = "./WikiSpeciesHabitats/"
        ## Area of interest
        studyArea = gpd.read_file("./raw_data/studyArea/studyArea.shp")
        ## Habitat maps processing
        hmp = HabitatMapsProcessor(studyArea=studyArea, cantonsList=["VS","VD"], rawDataPath=rawDataPath)
        habitatsMap, habitatsData = hmp.process_cantons()
        hmp = None
        ## Saving data
        habitatsMap.to_file(processedDataPath+"habitatsMap.gpkg", driver="GPKG")
        print("Saved habitatsMap.gpkg, head is the following")
        print(habitatsMap.head())
        habitatsData.to_json(finalDataPath+"habitatsData.json", orient="records")
        print("Saved habitatsData.json, head is the following")
        print(habitatsData.head())


class Step2():
    """Step 2 : Species records processing"""
    def __init__(self):
        print("Step 2 : Species records processing")
        ## Paths
        rawDataPath = "./raw_data/"
        processedDataPath = "./processed_data/"
        finalDataPath = "./WikiSpeciesHabitats/"
        ## Area of interest
        studyArea = gpd.read_file("./raw_data/studyArea/studyArea.shp")
        ## Species records processing
        srp = SpeciesRecordsProcessor(studyArea=studyArea, rawDataPath=rawDataPath, finalDataPath=finalDataPath)
        speciesRecords = srp.process_records()
        srp = None
        ## Saving data and unique species records
        speciesRecords.to_file(processedDataPath+"speciesRecords.gpkg", driver="GPKG")
        print("Saved speciesRecords.gpkg, head is the following")
        print(speciesRecords.head())
        fields = ["scientificName","kingdom","phylum","class","order","family","genus","species","speciesKey"]
        speciesData = speciesRecords[fields].drop_duplicates()
        speciesData.to_json(finalDataPath+"speciesData.json")
        print("Saved speciesData.json, head is the following")
        print(speciesData.head())


class Step3():
    """Step 3 : Intersecting species and records"""
    def __init__(self):
        print("Step 3 : Intersecting species and records")
        ## Paths
        processedDataPath = "./processed_data/"
        finalDataPath = "./WikiSpeciesHabitats/"
        ## Loading data
        habitatsMap = gpd.read_file(processedDataPath+"habitatsMap.gpkg")
        speciesRecords = gpd.read_file(processedDataPath+"speciesRecords.gpkg")
        ## Species-habitats pairs
        shm = SpeciesHabitatMerger()
        speciesHabitatsRecords = shm.merge_data(species=speciesRecords, habitats=habitatsMap)
        ## Saving species and habitats pairs
        speciesHabitatsRecords.to_json(processedDataPath+"speciesHabitatsRecords.json", orient="records")
        print("Saved speciesHabitatsRecords.json, head is the following")
        print(speciesHabitatsRecords.head())
        ## Saving species recorded in each zone
        speciesHabitatsRecords.groupby(["zoneID","TypoCH_NUM"])["speciesKey"].agg(["unique"]).reset_index().to_json(finalDataPath+"speciesInZones.json", orient="records")
        print("Saved speciesInZones.json, head is the following")
        print(speciesHabitatsRecords.groupby(["zoneID","TypoCH_NUM"])["speciesKey"].agg(["unique"]).reset_index().head())
        ## Saving species recorded in each habitat type
        speciesHabitatsRecords.groupby(["TypoCH_NUM"])["speciesKey"].agg(["unique"]).reset_index().to_json(processedDataPath+"speciesInHabitats.json", orient="records")
        print("Saved speciesInHabitats.json, head is the following")
        print(speciesHabitatsRecords.groupby(["TypoCH_NUM"])["speciesKey"].agg(["unique"]).reset_index())
        ## Saving habitats recorded for each species
        speciesHabitatsRecords.groupby(["speciesKey"])["TypoCH_NUM"].agg(["unique"]).reset_index().to_json(processedDataPath+"habitatsOfSpecies.json", orient="records")
        print("Saved habitatsOfSpecies.json, head is the following")
        print(speciesHabitatsRecords.groupby(["speciesKey"])["TypoCH_NUM"].agg(["unique"]).reset_index())




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





