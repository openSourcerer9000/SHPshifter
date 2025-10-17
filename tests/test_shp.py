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

    # @pytest.mark.parametrize("shuffle", [False,True] )
    @pytest.mark.parametrize("gdf, i, npts, expected", [
        (gpd.GeoDataFrame(geometry=[LineString([(x, y) for y in range(10)]) for x in range(6)]),
         0.5, 10, LineString([(2.5, y) for y in range(10)])),
        (gpd.GeoDataFrame(geometry=[LineString([(x, y) for y in range(10)]) for x in range(5)]),
         0, 10, LineString([(0, y) for y in range(10)])),
        (gpd.GeoDataFrame(geometry=[LineString([(x, y) for y in range(10)]) for x in range(5)]),
         1, 10, LineString([(4, y) for y in range(10)]))
    ])
    def test_interpLines(self,gdf, i, npts, expected):
        if len(gdf)==6:
        # if shuffle:
            idx = [0 , 3,5,4,2,1]
            idx = idx[:len(gdf)]
            gdf = gdf.reindex(idx)
            # gdf = gdf.reset_index(drop=True)
            # assert not gdf.isna().any().any()
        result = shp.gp.interpLines(gdf, i, npts)
        assert result.equals(expected), (result.wkt,'\n',expected.wkt)

    
    # def test_extractVerts(self):
    #     pts1 = [[0,0],[0,1],[1,1],[1,0]]

    #     gdf = gpd.GeoDataFrame({'col':['lulo','guanabana']},
    #         geometry=[Polygon(pts1),LineString(pts1)],crs='EPSG:4326')
    #     ptgdf = gdf.copy()
    #     ptgdf[g] = [[Point(i) for i in pts1]]*2
    #     ... how exactly it's gonna work? TODO

    #     assert_geodataframe_equal(extractVerts(gdf),ptgdf)

    import pytest

# Fixtures for sample data and buffer size
# @pytest.fixture
# def sample_dataset():
#     lat = np.linspace(0, 10, 11)  # Latitude from 0 to 10
#     lon = np.linspace(0, 20, 21)  # Longitude from 0 to 20
#     data = np.random.rand(len(lat), len(lon))  # Random data values
#     return xr.Dataset(
#         {
#             'data': (['latitude', 'longitude'], data)
#         },
#         coords={
#             'latitude': lat,
#             'longitude': lon
#         }
#     )

# @pytest.fixture
# def buffer_size():
    # return 5  # Degree buffer


# Fixtures for sample datasets
@pytest.fixture
def dataset_small():
    lat = np.linspace(0, 5, 6)  # Latitude from 0 to 5
    lon = np.linspace(0, 10, 11)  # Longitude from 0 to 10
    data = np.random.rand(len(lat), len(lon))  # Random data values
    return xr.Dataset(
        {
            'data': (['latitude', 'longitude'], data)
        },
        coords={
            'latitude': lat,
            'longitude': lon
        }
    )

@pytest.fixture
def dataset_large():
    lat = np.linspace(0, 50, 51)  # Latitude from 0 to 50
    lon = np.linspace(0, 100, 101)  # Longitude from 0 to 100
    data = np.random.rand(len(lat), len(lon))  # Random data values
    return xr.Dataset(
        {
            'data': (['latitude', 'longitude'], data)
        },
        coords={
            'latitude': lat,
            'longitude': lon
        }
    )

# Buffer sizes to test
buffer_sizes = [2, 5, 10, 15]
from itertools import product

# Parameterize using itertools.product to combine datasets and buffer sizes
@pytest.mark.parametrize("dataset, buffer_size", list(product(
    [pytest.lazy_fixture('dataset_small'), pytest.lazy_fixture('dataset_large')],
    buffer_sizes
)))
class TestBufferFunction:

    def test_buffer_expands_lat_lon(self, dataset, buffer_size):
        """Test that the buffer function expands latitude and longitude dimensions correctly."""
        ds_expanded = shp.nd.buffer(dataset, buffer_size)
        assert ds_expanded['latitude'].min() < dataset['latitude'].min(), "Latitude should expand with the buffer"
        assert ds_expanded['latitude'].max() > dataset['latitude'].max(), "Latitude should expand with the buffer"
        assert ds_expanded['longitude'].min() < dataset['longitude'].min(), "Longitude should expand with the buffer"
        assert ds_expanded['longitude'].max() > dataset['longitude'].max(), "Longitude should expand with the buffer"
    
    def test_original_data_preserved(self, dataset, buffer_size):
        """Test that original data points are preserved after the buffer expansion."""
        ds_expanded = shp.nd.buffer(dataset, buffer_size)
        original_data = dataset['data'].values
        expanded_data = ds_expanded['data'].sel(
            latitude=dataset['latitude'],
            longitude=dataset['longitude']
        ).values
        np.testing.assert_array_equal(original_data, expanded_data, "Original data points should be preserved in the expanded dataset")
    
    def test_new_data_filled_with_nan(self, dataset, buffer_size):
        """Test that new areas introduced by the buffer are filled with NaN."""
        ds_expanded = shp.nd.buffer(dataset, buffer_size)
        buffer_lat = np.setdiff1d(ds_expanded['latitude'], dataset['latitude'])
        buffer_lon = np.setdiff1d(ds_expanded['longitude'], dataset['longitude'])
        nan_data = ds_expanded['data'].sel(latitude=buffer_lat, longitude=buffer_lon)
        assert np.isnan(nan_data).all(), "Newly introduced values should be filled with NaN"
    
    def test_custom_fill_value(self, dataset, buffer_size):
        """Test that custom fill values are correctly applied."""
        custom_fill_value = -9999
        ds_expanded = shp.nd.buffer(dataset, buffer_size, fill_value=custom_fill_value)
        buffer_lat = np.setdiff1d(ds_expanded['latitude'], dataset['latitude'])
        buffer_lon = np.setdiff1d(ds_expanded['longitude'], dataset['longitude'])
        fill_data = ds_expanded['data'].sel(latitude=buffer_lat, longitude=buffer_lon)
        assert np.all(fill_data == custom_fill_value), "Newly introduced values should be filled with the custom fill value"

from shapely.geometry import box

@pytest.mark.parametrize("ds_bounds, gdf_bounds, expected_diff", [
    # Test case where ds fully contains gdf (all differences should be 0)
    ({"min_lon": -100, "max_lon": -90, "min_lat": 30, "max_lat": 40},  # ds bounds
        [-95, 32, -93, 35],  # gdf bounds [minx, miny, maxx, maxy]
        0),  # Expected difference
    
    # Test case where gdf extends beyond ds bounds on the min side (positive difference)
    ({"min_lon": -100, "max_lon": -90, "min_lat": 30, "max_lat": 40},
        [-105, 32, -93, 35],  # gdf extends beyond ds min longitude
        5),  # Expected positive difference (gdf minx extends by 5)
    
    # Test case where gdf extends beyond ds bounds on the max side
    ({"min_lon": -100, "max_lon": -90, "min_lat": 30, "max_lat": 40},
        [-95, 32, -85, 35],  # gdf extends beyond ds max longitude
        5),  # Expected positive difference (gdf maxx extends by 5)
    
    # Test case where gdf extends beyond ds bounds on both sides
    ({"min_lon": -100, "max_lon": -90, "min_lat": 30, "max_lat": 40},
        [-105, 28, -85, 42],  # gdf extends both min and max longitude/latitude
        5),  # Expected positive difference (max extension)
    
    # Test case where ds extends beyond gdf bounds (should return 0)
    ({"min_lon": -105, "max_lon": -85, "min_lat": 28, "max_lat": 50},
        [-100, 30, -85, 45],  # gdf is fully inside ds
        0),  # Expected difference is 0
])
def test_bound_diff(ds_bounds, gdf_bounds, expected_diff):
    # Create a mock xarray dataset
    lon = np.linspace(ds_bounds["min_lon"], ds_bounds["max_lon"], 10)
    lat = np.linspace(ds_bounds["min_lat"], ds_bounds["max_lat"], 10)
    ds = xr.Dataset(
        {
            "data": (("latitude", "longitude"), np.random.rand(10, 10)),
        },
        coords={
            "longitude": lon,
            "latitude": lat,
        }
    )

    # Create a mock GeoDataFrame using shapely's box function
    gdf = gpd.GeoDataFrame(geometry=[box(*gdf_bounds)], crs="EPSG:4326")

    # Call the function and check the result
    result = shp.nd.boundDiff(ds, gdf)
    assert result == expected_diff, f"Expected {expected_diff}, got {result}"
