## External packages
import pandas as pd
from tqdm import tqdm

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
            subset_habitats["Shape_Area"] = subset_habitats.area
            subset_species = subset_species.sjoin(subset_habitats, how="inner")
            ## Multiprocessing is useless with small batches like this
            #subset_species = dgpd.from_geopandas(subset_species, npartitions=8).sjoin(dgpd.from_geopandas(subset_habitats, npartitions=8), how="inner").compute()
            species_habitats_records = pd.concat([species_habitats_records, subset_species])
            species = species.drop(species[species["grid_id"]==i].index)
            habitats = habitats.drop(habitats[habitats["grid_id"]==i].index)
        return species_habitats_records