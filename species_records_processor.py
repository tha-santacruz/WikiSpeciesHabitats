## Standard Library
import os

## External packages
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from pyproj import Transformer

## Custom modules
from grid import GridBuilder



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
        print(records.columns)
        records = records.rename(columns={"speciesKey":"species_key","gbifID":"gbif_id","scientificName":"scientific_name",
                                          "decimalLatitude":"lat","decimalLongitude":"lon","eventDate":"event_date"})
        # Filtering on date
        records = records[records["year"]>=1950]
        # Filtering on taxonomy level
        records = records.dropna(subset=["species","species_key"])
        records["species_key"] = records["species_key"].apply(lambda x : int(x))
        # Keeping meaningful fields
        fields=["gbif_id", "scientific_name", "kingdom", "phylum", "class", "order", "family", 
                "genus", "species", "lat", "lon", "event_date", "species_key"]
        records = records[fields]
        return records

    def reproject(self, records, old="EPSG:4326", new="EPSG:2056"):
        """Reprojecting shape from old to new CRS"""
        ## Set transformation
        trans = Transformer.from_crs(old, new, always_xy=True)
        ## Reproject
        xx, yy = trans.transform(records["lon"].values, records["lat"].values)
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
        records = records[records["species_key"].isin(species_list)]
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