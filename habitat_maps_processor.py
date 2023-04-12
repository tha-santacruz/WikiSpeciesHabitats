## External packages
import dask_geopandas as dgpd
import pandas as pd
from tqdm import tqdm

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