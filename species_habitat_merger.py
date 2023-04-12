## Standard library
import warnings
warnings.filterwarnings('ignore')

## External packages
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from tqdm import tqdm, trange

## Custom packages
from grid import GridBuilder

class SpeciesHabitatMerger():
    """This class is intended to merge processed species and natural habitats data"""
    def __init__(self, study_area, processed_data_path):
        self.study_area = study_area
        self.processed_data_path = processed_data_path

    def merge_data(self, species, habitats):
        ## Performing inner spatial join
        #species_habitats_records = dgpd.from_geopandas(species, npartitions=8).sjoin(dgpd.from_geopandas(habitats, npartitions=8), how="inner").compute()
        ## Performing inner spatial join (with for loop to alleviate memory usage)
        species_habitats_records = pd.DataFrame()
        for i in tqdm(species["grid_id"].unique()):
            subset_species = species[species["grid_id"]==i]
            subset_habitats = habitats[habitats["grid_id"]==i]
            subset_habitats["Shape_Area"] = subset_habitats.area
            subset_species = subset_species.sjoin(subset_habitats, how="inner")
            ## Multiprocessing is useless with small batches like this
            #subset_species = dgpd.from_geopandas(subset_species, npartitions=8).sjoin(dgpd.from_geopandas(subset_habitats, npartitions=8), how="inner").compute()
            species_habitats_records = pd.concat([species_habitats_records, subset_species])
            species = species.drop(species[species["grid_id"]==i].index)
            habitats = habitats.drop(habitats[habitats["grid_id"]==i].index)
        return species_habitats_records
    
    def process_chunks(self):
        ## Grid
        grid = GridBuilder().grid_from_shape(shape=self.study_area, width=25000, height=25000)
        print("Defined chunks")
        ## Process by big chunks to alleviate memory usage
        species_habitats_records = pd.DataFrame()
        for i in trange(len(grid)):
            gridCell = grid.loc[i].geometry
            ## Loading data
            habitats_map = gpd.read_file(self.processed_data_path+"habitats_map.gpkg", mask=gridCell)
            species_records = gpd.read_file(self.processed_data_path+"species_records.gpkg", mask=gridCell)
            ## Species-habitats pairs
            newRecords = self.merge_data(species=species_records, habitats=habitats_map)
            species_habitats_records = pd.concat([species_habitats_records, newRecords])
        print("Intersected data")
        ## Keeping meaningful columns only
        species_habitats_records = species_habitats_records[["zone_id", "grid_id_right", "TypoCH_NUM", "species_key", "Shape_Area", "canton", "split"]]
        ## Renaming columns
        species_habitats_records = species_habitats_records.rename(columns={"grid_id_right":"grid_id", "Shape_Area": "shape_area"})
        print("Trimmed fields")
        return species_habitats_records