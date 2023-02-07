import pytest
from SHPshifter import SHPshifter as shp
from geopandas.testing import assert_geodataframe_equal
from pathlib import Path
import pandas as pd, numpy as np
import geopandas as gpd, xarray as xr
import rioxarray as rxr
import os
from shapely.geometry import LineString

# pytest test_SHPshifter.py -svv

mc = Path(os.path.dirname(os.path.abspath(__file__)))/'mockdata'
mcgis = mc/'GIS'

def test_cut():
    l = LineString([(0,0),(10,0)])
    assert shp.cut(l,.1,False)[0].length == 0.1
def test_cut_normalized():
    l = LineString([(0,0),(10,0)])
    assert shp.cut(l,.1,True)[0].length == 1
def test_cut_Z():
    l = LineString([(0,0,0),(10,10,10)])
    assert shp.cut(l,.5,True)[0].wkt == 'LINESTRING Z (0 0 0, 5 5 5)'
def test_cutPiece():
    l = LineString([(0,0),(10,0)])
    assert shp.cutPiece(l,.1,.85,True).wkt == 'LINESTRING (1 0, 8.5 0)'

def test_addZ():
    line = LineString([(0,0),(1,1)])
    Z = [5,10]
    assert shp.addZ(line,Z).wkt=='LINESTRING Z (0 0 5, 1 1 10)'
def test_addZ_alreadyZ():
    line = LineString([(0,0,0),(1,1,1)])
    Z = [5,10]
    assert shp.addZ(line,Z).wkt=='LINESTRING Z (0 0 5, 1 1 10)'
def test_addZ_already_array():
    line = LineString([(0,0),(1,1)])
    Z = [5,10]
    Z = np.array([Z]).T
    assert shp.addZ(line,Z).wkt=='LINESTRING Z (0 0 5, 1 1 10)'

class Testgp:
    def test_concat_gdflist(self):
        gdfs = [shp.gp.randPts(20,'GDF',seedx=seed) for seed in [1,2,3] ]
        assert isinstance(gdfs[0],gpd.GeoDataFrame)
        assert gdfs[0].crs
        cat = shp.gp.concat(gdfs)
        assert isinstance(cat,gpd.GeoDataFrame)
        assert cat.crs==gdfs[0].crs
        assert len(cat)==60
    def test_concat_geoserieslist(self):
        gss = [shp.gp.randPts(20,seedx=seed) for seed in [4,5,6] ]
        assert isinstance(gss[0],gpd.GeoSeries)
        assert gss[0].crs
        cat = shp.gp.concat(gss)
        assert isinstance(cat,gpd.GeoSeries)
        assert cat.crs==gss[0].crs
        assert len(cat)==60
    def test_asGDF(self):
        shpth = mcgis/'stream'/'onegood'/f'goodstream.geojson'
        gdf = gpd.read_file(shpth)
        fromshp = shp.gp.asGDF(shpth)
        fromgdf = shp.gp.asGDF(gdf)
        assert isinstance(fromgdf,gpd.GeoDataFrame)
        assert isinstance(fromshp,gpd.GeoDataFrame)
    def test_asGDF_kwargs(self):
        shpth = mcgis/'stream'/'onegood'/f'goodstream.geojson'
        gdf = gpd.read_file(shpth)
        fromshp = shp.gp.asGDF(shpth,rows=1)
        fromgdf = shp.gp.asGDF(gdf,rows=1)
        assert isinstance(fromgdf,gpd.GeoDataFrame)
        assert isinstance(fromshp,gpd.GeoDataFrame)
        assert len(fromgdf)>1 #just a requirement for the shp in shpth for this test to make sense
        assert len(fromshp)==1
    def test_joinAttrsByNearest(self):
        #example came up with that only works recursively, open in GIS to see geo
        boxvec = mcgis/'joinAttrs'/'boxes.geojson'
        linevec = mcgis/'joinAttrs'/'linez.geojson'
        joind = shp.gp.joinAttrsByNearest(boxvec,linevec,
            maxdist=25/2,recursive=True)

        assert joind['name'].str[-1].to_list() == joind['linee'].str[-1].to_list()
    
    @pytest.mark.parametrize('subdir',
        ['allbad',
        'onegood']
    )
    def test_fixMultiLineStrings(self,subdir):
        good,bad = [ gpd.read_file(mcgis/'stream'/subdir/f'{nm}stream.geojson')
            for nm in ('good','bad') ]
        
        assert shp.gp.fixMultiLineStrings(bad).crs==good.crs
        assert_geodataframe_equal(
            shp.gp.fixMultiLineStrings(bad) , good,
            check_dtype=False,
            check_crs=False,
            check_less_precise=True
            )
    
    # def test_extractVerts(self):
    #     pts1 = [[0,0],[0,1],[1,1],[1,0]]

    #     gdf = gpd.GeoDataFrame({'col':['lulo','guanabana']},
    #         geometry=[Polygon(pts1),LineString(pts1)],crs='EPSG:4326')
    #     ptgdf = gdf.copy()
    #     ptgdf[g] = [[Point(i) for i in pts1]]*2
    #     ... how exactly it's gonna work? TODO

    #     assert_geodataframe_equal(extractVerts(gdf),ptgdf)