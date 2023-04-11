import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np

class GridBuilder():
    """
    Allows to build a recutangular grid overlapping an interest zone with desired grid size
    Interest zone can be determined either using coordinates or a GeoDataFrame
    Grid is returned as a GeoDataFrame
    Code from https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
    """
    def __init__(self):
        pass

    def split_cells(self, shape, num_stripes=5):
        shape["split"] = "train"
        ids = shape.index.to_list()
        splitted = np.array_split(ids, num_stripes)
        for i in range(num_stripes):
            splitted[i] = np.array_split(splitted[i],10)
            for j in range(10):
                if j > 3:
                    shape.loc[splitted[i][j],"split"] = "train"
                elif j > 0:
                    shape.loc[splitted[i][j],"split"] = "test"
                else:
                    shape.loc[splitted[i][j],"split"] = "val"
        return shape


    def grid_from_shape(self, shape, width=1000, height=1000, split=True, num_stripes=5):
        """Create a grid from reference shape and crs"""
        ## Get reference shape bounds and crs
        xmin, ymin, xmax, ymax =  shape.total_bounds
        crs = shape.crs

        ## Compute number of rows and colums needed
        rows = int(np.ceil((ymax-ymin) /  height))
        cols = int(np.ceil((xmax-xmin) / width))

        ## Initialize corners
        XleftOrigin = xmin
        XrightOrigin = xmin + width
        YtopOrigin = ymax
        YbottomOrigin = ymax- height

        ## Iteratively build polygons
        polygons = []
        for i in range(cols):
            Ytop = YtopOrigin
            Ybottom =YbottomOrigin
            for j in range(rows):
                polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
                Ytop = Ytop - height
                Ybottom = Ybottom - height
            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width

        ## Save polygons as a GeoDataFrame, assign CRS and return
        grid = gpd.GeoDataFrame({'geometry':polygons})
        grid = grid.set_crs(crs=crs)

        ## Assign split if requested
        if split:
            grid = self.split_cells(grid, num_stripes=num_stripes)
        return grid
    
    def grid_from_points(self, xmin, ymin, xmax, ymax, crs, width=1000, height=1000, split=True):
        """Create a grid from point coordinates and provided crs"""
        ## Compute number of rows and colums needed
        rows = int(np.ceil((ymax-ymin) /  height))
        cols = int(np.ceil((xmax-xmin) / width))

        ## Initialize corners
        XleftOrigin = xmin
        XrightOrigin = xmin + width
        YtopOrigin = ymax
        YbottomOrigin = ymax- height

        ## Iteratively build polygons
        polygons = []
        for i in range(cols):
            Ytop = YtopOrigin
            Ybottom =YbottomOrigin
            for j in range(rows):
                polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
                Ytop = Ytop - height
                Ybottom = Ybottom - height
            XleftOrigin = XleftOrigin + width
            XrightOrigin = XrightOrigin + width

        ## Save polygons as a GeoDataFrame, assign CRS and return
        grid = gpd.GeoDataFrame({'geometry':polygons})
        grid = grid.set_crs(crs=crs)

        ## Assign split if requested
        if split:
            grid = self.split_cells(grid)
        return grid
    
if __name__ == "__main__":
    shape = gpd.read_file("raw_data/studyArea/studyArea.shp")
    bg = GridBuilder()
    grid = bg.grid_from_shape(shape=shape)
    grid.to_file("grid.shp")