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

    def grid_from_shape(self, shape, width=1000, height=1000):
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
        return grid
    
    def grid_from_points(self, xmin, ymin, xmax, ymax, crs, width=1000, height=1000):
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
        return grid