from funkshuns import *
    
from pathlib import Path
import pyproj

GISdata = Path('GISdata')
printImportWarnings = False

g = 'geometry'
wgs84='EPSG:4326'

try:
    import xarray as xr
    import rioxarray as rxr
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)
from shapely.geometry import Point, LineString, mapping, MultiPoint, Polygon, box, shape
from shapely import wkt
import shapely

from shapely.ops import substring, nearest_points
from functools import partial

reverse = partial(substring, start_dist=1, end_dist=0, normalized=True)
def cut(line, distance,normalized=False):
    ''' Cuts a line in two at a distance from its starting point\n
    returns [segment0,semgent1]\n\n
    if distance <= 0.0 or distance >= line.length:\n
        return [LineString(line)]'''
    if normalized:
        distance=distance*line.length
    if distance==0.0:
        return [None,LineString(line)]
    if distance==line.length:
        return [LineString(line),None]
    if distance < 0.0 or distance > line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            #cpt is [(cp.x, cp.y)] or [(cp.x, cp.y, cp.z)] etc depending on dims
            cpt = list(cp.coords) 
            return [
                LineString(coords[:i] + cpt),
                LineString(cpt + coords[i:])]
def cutPiece(line,distance1, distance2,normalized=False):
    """ From a linestring, this cuts a piece of length lgth at distance.\n
    ie cutPiece(line,.1,.85,True) will cut a 10-85 from a LFP\n
    ex:\n
    gdf[g]=gdf[g].map(lambda line:shp.cutPiece(line,.1,.85,True))
    """
    if normalized:
        distance1,distance2 = distance1*line.length,distance2*line.length
    l1 = cut(line, distance1)[1]
    result = cut(l1, distance2-distance1)[0]
    # l2 = cut(line, distance2)[0]
    # result = l1.intersection(l2)
    assert isinstance(result,LineString), f'INTERNAL ERROR: {result} is not LineString, problem with cutPiece function'
    return result
def redistributeVertices(geom, distance):
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
            for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistributeVertices(part, distance)
                for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))
def addZ(line,Z):
    '''returns LineString Z with Z value from an iterable Z'''
    coordz = np.array(LinetoList(line))

    z = 'banana'
    #case if Z is already a proper np array:
    if isinstance(Z,np.ndarray):
        try:
            if Z.shape[1]==1:
                z = Z
        except:
            pass
    if not isinstance(z,np.ndarray): # not assigned in the special case above
        z = np.array([Z]).T

    c3 = np.concatenate([coordz,z],axis=1)
    return LineString(c3)
def LinetoList(myLineString,keepZ=False,emptyIsOK=True,noneIsOK=False):
    '''to ((x1,y1),(x2,y2),...)'''
    if myLineString is None:
        if noneIsOK:
            print('WARNING, no LineString passed to LinetoList(myLineString=None), returning []')
            return []
        else:
            assert False, 'ERROR, no LineString passed to LinetoList(myLineString=None)'
    if myLineString.is_empty:
        if emptyIsOK:
            return []
        else:
            assert False, f'ERROR: {myLineString} is empty.\nIf you would like to suppress this error and return [] instead, set emptyIsOK to True'
    xy = mapping(myLineString)['coordinates']
    if not keepZ and len(xy[0])==3: #has Z
        xy = [j[:2] for j in xy] 
    return xy
# from osgeo import ogr
from shapely.wkt import loads
# def shpDensify(geom,maxVertexDist):
#     '''gdf['geometry'] = gdf['geometry'].map(shpDensify,2)'''
#     wkt = geom.wkt  # shapely Polygon to wkt
#     geom = ogr.CreateGeometryFromWkt(wkt)  # create ogr geometry
#     geom.Segmentize(maxVertexDist)  # densify geometry
#     wkt2 = geom.ExportToWkt()  # ogr geometry to wkt
#     new = loads(wkt2)  # wkt to shapely Polygon
#     return new
def filterPts(line,n,byDist=False):
    '''filters shapely linestring line to n pts\n
    OR if byDist: by distance n
    \nTODO issue with returning duplicate points w/ byDist? use .simplify(0)?'''
    if byDist:
        res = [line.interpolate(n*i) for i in range(int(line.length/n)+1)]
        res += [line.interpolate(1,normalized=True)]
        res = LineString(res)
        return res
    else:
        return LineString( 
            [line.interpolate(i/(n-1),normalized=True) for i in range(n)]
            )
# def substring(geom, start_dist, end_dist, normalized=False):
#     """available in shapely 1.7! shapely.ops.substring()\n
#     Return a line segment between specified distances along a linear geometry.
#     Negative distance values are taken as measured in the reverse
#     direction from the end of the geometry. Out-of-range index
#     values are handled by clamping them to the valid range of values.
#     If the start distances equals the end distance, a point is being returned.
#     If the normalized arg is True, the distance will be interpreted as a
#     fraction of the geometry's length.
#     """
#     assert(isinstance(geom, LineString))
#     # Filter out cases in which to return a point
#     if start_dist == end_dist:
#         return geom.interpolate(start_dist, normalized)
#     elif not normalized and start_dist >= geom.length and end_dist >= geom.length:
#         return geom.interpolate(geom.length, normalized)
#     elif not normalized and -start_dist >= geom.length and -end_dist >= geom.length:
#         return geom.interpolate(0, normalized)
#     elif normalized and start_dist >= 1 and end_dist >= 1:
#         return geom.interpolate(1, normalized)
#     elif normalized and -start_dist >= 1 and -end_dist >= 1:
#         return geom.interpolate(0, normalized)
#     start_point = geom.interpolate(start_dist, normalized)
#     end_point = geom.interpolate(end_dist, normalized)
#     min_dist = min(start_dist, end_dist)
#     max_dist = max(start_dist, end_dist)
#     if normalized:
#         min_dist *= geom.length
#         max_dist *= geom.length
#     if start_dist < end_dist:
#         vertex_list = [(start_point.x, start_point.y)]
#     else:
#         vertex_list = [(end_point.x, end_point.y)]
#     coords = list(geom.coords)
#     current_distance = 0
#     for p1, p2 in zip(coords, coords[1:]):
#         if min_dist < current_distance < max_dist:
#             vertex_list.append(p1)
#         elif current_distance >= max_dist:
#             break
#         current_distance += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
#     if start_dist < end_dist:
#         vertex_list.append((end_point.x, end_point.y))
#     else:
#         vertex_list.append((start_point.x, start_point.y))
#         # reverse direction result
#         vertex_list = reversed(vertex_list)
#     return LineString(vertex_list)

def projectAll(geom,projectToGeom,normalized=False):
    '''shapely project for all pts in geom\n
    projectToGeom.project(geomPT0) for all pts'''
    projek = lambda pt:projectToGeom.project(Point(pt),normalized=normalized)
    if geom.type=='Polygon':
        geom=geom.boundary
    if geom.type=='LineString':
        STAs = [ projek(pt) for pt in geom.coords]
    else:
        raise NotImplementedError(f'geom type {geom.type} not supported')
    return STAs

import geopandas as gpd
try:
    import rasterio
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)

try:
    from scipy.spatial import cKDTree
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)
from shapely.geometry import Point
try:
    import topojson as tp
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)

from shapely import wkb
def dropZ(geom,output_dimension=2):
    '''drops Z or more dims from shapely geo'''
    return wkb.loads(wkb.dumps(geom, output_dimension=output_dimension))
from shapely.ops import linemerge
from geopandas.testing import assert_geodataframe_equal


class gp():
    '''geopandas widgets\n
    multipart to singlepart gdf.explode()'''
    def asGDF(GDF_or_Path,**kwargs):
        if isinstance(GDF_or_Path,PurePath):
            # if GDF_or_Path.suffix=='.shp':
            return gpd.read_file(GDF_or_Path,**kwargs)
        else:
            return GDF_or_Path.copy()
    def stillGDF(hopefullyGDF):
        assert isinstance(hopefullyGDF,gpd.GeoDataFrame)
        assert hopefullyGDF.crs
    def assertGDFequal(GDF_A,GDF_B,colOrderIrrelevent=True,**kwargs):
        '''GDF or vec file'''
        gdfA,gdfB = gp.asGDF(GDF_A),gp.asGDF(GDF_B)
        if colOrderIrrelevent:
            assert set(gdfB.columns)==set(gdfA.columns)
            gdfB = gdfB[gdfA.columns]
        assert_geodataframe_equal(gdfA,gdfB,**kwargs)
    def mingle(gpdf):
        '''multipart to singlepart'''
        gpdf_multipoly = gpdf[gpdf.geometry.type.str.startswith('Multi')]
        gpdf_singlepoly = gpdf[~gpdf.geometry.type.str.startswith('Multi')]
        
        # gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
        # gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

        for i, row in gpdf_multipoly.iterrows():
            Series_geometries = pd.Series(row.geometry)
            df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]*len(Series_geometries), ignore_index=True)
            df['geometry']  = Series_geometries
            gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

        gpdf_singlepoly.reset_index(inplace=True, drop=True)
        return gpdf_singlepoly
    def bbox(gdf,ret='box'):
        '''finds bbox of gdf (or geoseries)
        \nreturns shapely.geometry.box if ret=='box', GDF if ret=='gdf', or geoseries otherwise'''
        GDF = gpd.GeoDataFrame(geometry=gdf,crs=gdf.crs) if isinstance(gdf,gpd.GeoSeries) else gdf.copy()
        bnds = list(GDF.dissolve().bounds.iloc[0])
        bbx = box(*bnds)
        if ret.lower()=='box':
            return bbx
        elif ret.lower()=='gdf':
            return gpd.GeoDataFrame(geometry=[bbx],crs=gdf.crs)
        else:
            return gpd.GeoSeries([bbx],crs=gdf.crs)
    @timeit
    def voronoi(pts,perim):
        '''
        returns pts Point GDF with geom replaced by voronoi polygons, bounded by perim polygon/multipolygon GDF\n
        inputs as vec Path or GDF's TODO np array, shapely polygon
        '''
        assert int(shapely.__version__[0])>=2, f'Shapely v2.0 or greater required to use voronoi polygons, please upgrade.\nYour envs version: {shapely.__version__}'
        coordz = gp.asGDF(pts)
        perimm = gp.asGDF(perim).to_crs(coordz.crs).dissolve()
        
        ptz = coordz[g].to_list()
        mpt = MultiPoint(ptz)

        voro = shapely.voronoi_polygons(mpt
            # ,extend_to=perim[g].iloc[0] DON'T WORK
            )
        del ptz,mpt

        coordz[g]=voro.geoms
        del voro
        assert (coordz.type=='Polygon').all()
        
        cellz = coordz.clip(perimm)
        return cellz
    def dropZ(gdf,output_dimension=2):
        gdf[g] = gdf[g].map(lambda geom:dropZ(geom,output_dimension=output_dimension))
        return gdf
    def toGeoJSON(vecpth,outGeoJSON,simplifyFt=70,quant=0.1,preserveTopo=False):
        '''For webmappin\n
        attempts to convert simplifyFt to meters or deg as needed, and simplifies vertices\n
        converts to WGS 84\n
        vecpth or GDF to outGeoJSON\n
        returns the new GDF
        '''

        gdf = gp.asGDF(vecpth)
        unitz = gp.getUnits(gdf.crs)
        
        simp = simplifyFt
        if 'met' in unitz:
            simp *= 0.3048
        elif 'deg' in unitz:
            simp /= 364567.2

        if not preserveTopo:
            gdf[g] = gdf.simplify(simp)
            gdf=gdf.to_crs(wgs84)
            gdf.to_file(outGeoJSON,driver='GeoJSON')
        else:
            gdf = gdf.to_crs('EPSG:4326')
            simp = simplifyFt/364567.2

            topo = tp.Topology(gdf, simplify_algorithm='vw', toposimplify=simp)#, topoquantize=0.00001)
            topo.to_geojson(outGeoJSON)
        print(f'Converted to {outGeoJSON}')
        return gdf
    def to_json(GDF):
        '''
        returns serialized geojson str of GDF which can be parsed with gpd.read_file(geojsonString) \n
        built in method .to_json comes out messed up after roundtripping???\n'''
        tmpjson = Path.home()/'tmp.geojson'
        GDF.to_file(tmpjson,driver='GeoJSON')
        with open(tmpjson) as fh:
            jsn = json.load(fh)
        return json.dumps(jsn)
    def getUnits(coors):
        '''crs: pyproj.CRS (obtained w/ GDF.crs) or GDF or vec pth\n
        returns horizontal length unit in {'US survey foot',}'''
        if not isinstance(coors,pyproj.CRS):
            crrs = gp.asGDF(coors).crs
        else:
            crrs = coors
        return crrs.axis_info[0].unit_name
    def to_coors(badGDF,newCoordsys):
        '''no longer needed as of gpd 0.9\n
        #why you need to do this, I absolutely don't know
        #https://stackoverflow.com/questions/55390492/runtimeeror-bno-arguments-in-initialization-list'''
        goodgdf = gpd.GeoDataFrame(badGDF, geometry=badGDF['geometry'])
        goodgdf.crs = {'init':badGDF.crs['init']}
        goodgdf = goodgdf.to_crs({'init' :newCoordsys}) 
        return goodgdf
    def getSTA(GDF,joyn=False,nme='STA'):
        '''returns STA's [0,7,93.6,...,CL.length] along LineStrings as pd Series (retains (multi)index) \n
        GDF = geodataframe / Geoseries\n
        if joyn: returns latsDF with a new column [nme]'''
        if type(GDF) == gpd.GeoDataFrame:
            CLs = GDF['geometry']
        else:
            CLs = GDF
        dists = CLs.apply(lambda CLine : [CLine.project(Point(coord)) for coord in CLine.coords])
        # integration test, then DEL ALL THIS commented:
            #     dists = CLs.apply(lambda CLine : CLine)
            # dists = pd.DataFrame(columns=['River','Reach','RS','STA'])
            # i=0
            # for river,_ in CLs.groupby(level=0):
            #     for reach,_ in CLs.groupby(level=1):
            #         for RS,CL in CLs.groupby(level=2):
            #             CLine = CL.iloc[0]
            #             dists.loc[i] = [ river,reach,RS,[CLine.project(Point(coord)) for coord in CLine.coords] ]
            #             ##dists.update([( RS,[CLine.project(Point(coord)) for coord in CLine.coords] )])
            #             i+=1
            # dists=pd.DataFrame(dists)#,columns = ['STA'])
            # dists = dists.set_index(['River','Reach','RS'])
        dists.name = nme
        if joyn:
            dists = GDF.join(dists)
        return dists
    def FilterPts(linegdf,n,filterbyattr=None,outShapefile=None):
        '''edits inLines <inplace??? not used like that below> (Path obj -or- geodataframe/geoseries) of linestrings to have n pts \n
        and writes to outShapefile if specified\n
        Returns geodataframe \n
        filterbyattr = expression to evaluate as str: ex: "['Type']=='Lateral'" TODO (UPGRADE THAT WITH PD QUERY)
        '''

        if isinstance(linegdf, PurePath):
            # if inLines.suffix=='.shp':
            linegdf = gpd.read_file(linegdf)
        inLines = linegdf.copy()
        isGDF = type(inLines) == gpd.GeoDataFrame
        if filterbyattr:
            inLines = inLines[eval('inLines' + filterbyattr)]
        if isGDF:
            inLines['geometry'] = inLines['geometry'].apply(filterPts, args=[n])
            qc = inLines['geometry']
        else:
            assert type(inLines) == gpd.GeoSeries
            inLines= inLines.apply(filterPts, args=[n])
            qc = inLines
        #QC:
        qc = qc.apply(lambda x: len(x.coords))
        assert qc.min() == qc.max() == n
        #
        if outShapefile:
            inLines.to_file(outShapefile)
        return inLines
    # def extractVerts(gdf):
    #     '''
    #     Extracts vertices of 
            # TODO mirror the QGIS behavior?
    #     '''
    #     xy = np.array(gdf[g].iloc[0].coords)
    #     xy
    #     pts = gpd.GeoSeries(gpd.points_from_xy(xy[:,0],xy[:,1]),crs=gdf.crs)
    #     return pts
    def geoFromXY(DF,Xcol,Ycol):
        '''TODO this is already a gpd method??\n
        Converts DF cols 'Xcol' and 'Ycol' (columns are as lists of floats [x1,x2,x3,...])\n
        into a PD Series of shapely linestrings\n
        CAREFUL: Xcol Ycol cannot have special characters to use the method df.Xcol\n
        examples:\n
        myDF['geometry'] = gp.geoFromXY(myDF,'x','y')\n
        myGeo = gpd.GeoSeries( gp.geoFromXY(...) )'''
        #DFcols = DF.columns
        #REPLACE THIS with a general .replace that includes all illegal characters for the DF.Xcol form:
        DF.columns = [st.replace(' ','_').replace('@','_') for st in DF.columns]
        Xcol = Xcol.replace(' ','_').replace('@','_')
        Ycol = Ycol.replace(' ','_').replace('@','_')
        zyp = 'zip(df.' + Xcol + ',df.' + Ycol + ')'
        profSeries = DF.apply(lambda df: LineString(eval(zyp)),axis=1)
        return profSeries
    def XYFromGeo(seriez,Xname='X',Yname='Y'):
        '''returns a tuple of pd Series' (Xseries,Yseries), 
        with the format of lists of floats [x1,x2,x3,...] \n
        seriez: GeoSeries or column of shapely linestrings to convert\n
        Xname,Yname = str names for Xseries,Yseries\n
        exs:\n
        DF['STA'],DF['Z'] = gp.XYFromGeo(DF['Prof'])\n
        pd.DataFrame(gp.XYFromGeo(DF['Prof'])).T'''
        Xseries = seriez.map(lambda prof: [pt[0] for pt in LinetoList(prof)])
        Yseries = seriez.map(lambda prof: [pt[1] for pt in LinetoList(prof)])
        Xseries.name,Yseries.name = Xname,Yname
        return (Xseries,Yseries)
    def XYFromPts(GDF_pts,Xname='X',Yname='Y'):
        '''GDF_pts: geoseries or geodataframe of shapely POINTs\n
        ex:\n
        wsel_kep['Lat'],wsel_kep['Lon'] = XYFromPts(wsel_kep)'''
        if type(GDF_pts) == gpd.GeoDataFrame:
            pts = GDF_pts['geometry']
        else:
            pts = GDF_pts
        Xseries = pts.map(lambda p: p.x)
        Yseries = pts.map(lambda p: p.y)
        Xseries.name,Yseries.name = Xname,Yname
        return (Xseries,Yseries)
    def gdfFromFrame(df,dfXY,coordsys=None,dropXYcols=True):
        '''returns a geodataframe from a pd df and a dfXY of the same length whose columns are X,Y'''
        DF= df.copy()
        DF.index = pd.MultiIndex.from_frame(dfXY)
        DF = DF.reset_index()
        # print (gpd.GeoDataFrame(DF, geometry=gpd.points_from_xy(DF.X,DF.Y),crs = coordsys))
        gdf = gpd.GeoDataFrame(DF, geometry=gpd.points_from_xy(DF.X,DF.Y))
        if coordsys:
            gdf.crs = coordsys
        if dropXYcols:
            gdf = gdf.drop(['X','Y'],axis=1)
        return gdf
    def concat(geoSerieses,**kwargs):
        '''concats without losing CRS\n
        only tested for geoseries for now TODO gdfList'''
        geoz = [gg for gg in geoSerieses if gg is not None]
        tstdiff = len(geoSerieses)-len(geoz)
        if tstdiff:
            print(f'Warning, {tstdiff} items not found in\n {geoSerieses}')
            
        coors = geoz[0].crs
        def crusade(ser):
            if ser.crs!=coors:
                ser = ser.to_crs(coors)
            return ser
        geoz = [crusade(ser) for ser in geoz]

        for ser in geoz[1:]:
            assert ser.crs==coors, 'Coordinate systems do not line up'
            
        # print('kwargs',kwargs)
        isSeries = isinstance(geoz[0],gpd.geoseries.GeoSeries) #TODO should check any or all?
        cat = pd.concat(geoz,**kwargs,   \
            ignore_index=isSeries)
        cat.name='geometry'
        return gpd.GeoSeries(cat, crs=coors) if isSeries else gpd.GeoDataFrame(cat, crs=coors)
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    def nearest(gdfA, gdfB,k=1,**queryKwargs):
        '''returns GeoDataFrame of \n
        k: # of nearest neighbors\n
        if CRS doesn't match, uses gdfB.to_crs(gdfA.crs) (doesn't actually change gdfB)\n
        TODO fails if index(es) isn't range(len(gdA)) , fix this \n
        #TODO use centroid for polygons
        queryKwargs get passed to KDTree.query\n
        ie distance_upper_bound=10\n
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html
        '''
        gdA = gdfA
        gdB = gdfB if (gdfA.crs==gdfB.crs) else gdfB.to_crs(gdfA.crs)

        nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=k,**queryKwargs)
        if k==1:
            gdf = pd.concat(
                [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
                    pd.Series(dist, name='dist')], axis=1)
        else:
            alldfs = [ pd.concat(
                [gdA.reset_index(drop=True),gdB.loc[idx[:,col], gdB.columns != 'geometry'].reset_index(drop=True),
                    pd.Series(dist[:,col], name='dist')], axis=1)  for col in range(idx.shape[1]) ]
            gdf = pd.concat(alldfs,axis=0) #TODO should work gp.concat(alldfs,axis=0)
        return gdf
    def joinAttrsByNearest(gdfA,gdfB,maxdist=1000,recursive=False,maxtries=10):
        '''Join attrs from gdf_B to gdf_A one-to-one by "nearest" up to maxdist\n
        if no neighbors found within maxdist for a feature, the feature's attrs will be NaN\n
        datatypes can be whatever\n
        if recursive: will start with maxdist, then double it until a match is found for all features\n
        or maxTries exceeded
        '''
        gdf_A,gdf_B = gp.asGDF(gdfA),gp.asGDF(gdfB)

        gdf_A['_idx_'] = range(len(gdf_A))
        
        B = gdf_B.copy()
        B = B.to_crs(gdf_A.crs)
        B[g] = B.buffer(maxdist)

        xx = gpd.overlay(gdf_A,B,keep_geom_type=False)
        if xx.type.iloc[0]=='Polygon':
            xx['weightt']=xx.area
        else: #linestring
            xx['weightt']=xx.length
        xx=xx.sort_values('weightt',ascending=False)
        xx

        xx=xx.groupby('_idx_').first().reset_index()
        # xx=xx.drop(['_idx_','weightt'],axis=1)
        xx
        assert len(xx)<=len(gdf_A)
        attrz = xx.columns[~xx.columns.isin(gdf_A.columns)].drop('weightt').to_list()
        attrz
        xx[attrz+['_idx_']]
        mergd = gdf_A.merge(xx[attrz+['_idx_']],on='_idx_',how='left')
        mergd
        mergd = mergd.drop('_idx_',axis=1)

        #clean up gdf A too as it added col to OG 
        gdf_A = gdf_A.drop('_idx_',axis=1)
        del B

        if recursive:
            joind = mergd

            newcols = gdf_B.columns[~gdf_B.columns.isin(gdf_A.columns)]
            newcol=newcols[0]
            newcols
            assert newcol in joind.columns and newcol not in gdf_A.columns
            newcol
            
            i=2
            while joind[newcol].isna().any():
                if i>2**maxtries-1:
                    raise f'max tries exceeded. Increase maxdist from {maxdist} or check your projection units'
                print(f'expanding maxdist to {maxdist*i}')
                nu = gp.joinAttrsByNearest(gdf_A,gdf_B,maxdist*i)
                
                joind[newcols] = joind[newcols].fillna(nu[newcols])
                i *= 2
            mergd = joind
        return mergd
    def extractVerts(lineGeoSer):
        ''''''
        verts = lineGeoSer.apply(lambda ln: [Point(pt) for pt in ln.coords])
        verts = gp.concat([ gpd.GeoSeries(pts) for pts in verts ])
        return verts
    def drape(gdf,demtif):
        '''

        '''
        dem = rast.asRioxds(demtif)
        Z = rast.sample(gdf,dem)

        if gdf[g].type.iloc[0]=='LineString':
            lines = gdf
            lines['_zeee'] = Z
            lines[g] = lines.apply(lambda line:
                # line['_zeee']
                addZ(line[g],line['_zeee'])
                ,axis=1)
            return lines.drop('_zeee',axis=1)
        else:
            raise NotImplementedError(f'geom of type {gdf[g].type.iloc[0]} not yet supported for shp.gp.drape')
    def getZ(x,aggFunc=False):
        '''gdf['z'] = gdf.geometry.apply(gp.getZ)
        use exterior and interior methods to extract the coordinates as tuples (x,y,z). \n
        I belive this should work for both polygon and multipolygons with any number of interior rings\n
        if aggFunc: apply aggFunc(list of Z's) per feature \n
        ex.\n gdf['z'] = gdf[g].apply(gp.getZ,aggFunc=max)\n
        https://gis.stackexchange.com/questions/333327/getting-z-coordinates-from-polygon-list-of-geodataframe'''
        if x.type in ['Polygon','LineString']:
            x = [x]
        zlist = []

        for shpp in x:
            if shpp.type == 'Polygon':
                zlist.extend([c[-1] for c in shpp.exterior.coords[:-1]])
                for inner_ring in shpp.interiors:
                    zlist.extend([c[-1] for c in inner_ring.coords[:-1]])
            else: #linestring assumed
                zlist.extend([c[-1] for c in shpp.coords[:-1]])
        if aggFunc:
            zlist = aggFunc(zlist)
        return zlist
        #return sum(zlist)/len(zlist) #In your case to get mean. Or just return zlist[0] if they are all the same
    def remDups(vectorPath,dupCol,sufix='B',_da=None):
        '''Duplicates in col {dupCol} appended with {sufix} and overwritten to {vectorPath}'''
        dapth = vectorPath
        if _da is not None:
            da = _da
        else:
            da = gp.asGDF(dapth)
        da
        
        dex = da[da.DA.duplicated()].index
        dex
        
        da.loc[dex,dupCol] = da.loc[dex,dupCol].map(lambda st: f'{st}{sufix}')
        da.loc[dex,dupCol]
        
        if len(da[da.DA.duplicated()])==0:
            # if isinstance(dapth,PurePath): #top recursion lvl
            fileBackup(dapth)
            da.to_file(dapth)
            print(f'Duplicates in col {dupCol} appended with {sufix} and saved to {dapth}')
        else:
            gp.remDups(dapth,dupCol,sufix,_da=da)
        

    #https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
    # from sklearn.neighbors import BallTree
    # import numpy as np

    # def get_nearest(src_points, candidates, k_neighbors=1):
    #     """Find nearest neighbors for all source points from a set of candidate points"""

    #     # Create tree from the candidate points
    #     tree = BallTree(candidates, leaf_size=15, metric='haversine')

    #     # Find closest points and distances
    #     distances, indices = tree.query(src_points, k=k_neighbors)

    #     # Transpose to get distances and indices into arrays
    #     distances = distances.transpose()
    #     indices = indices.transpose()

    #     # Get closest indices and distances (i.e. array at index 0)
    #     # note: for the second closest points, you would take index 1, etc.
    #     closest = indices[0]
    #     closest_dist = distances[0]

    #     # Return indices and distances
    #     return (closest, closest_dist)


    # def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    #     """
    #     For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    #     NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    #     """

    #     left_geom_col = left_gdf.geometry.name
    #     right_geom_col = right_gdf.geometry.name

    #     # Ensure that index in right gdf is formed of sequential numbers
    #     right = right_gdf.copy().reset_index(drop=True)

    #     # Parse coordinates from points and insert them into a numpy array as RADIANS
    #     left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    #     right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    #     # Find the nearest points
    #     # -----------------------
    #     # closest ==> index in right_gdf that corresponds to the closest point
    #     # dist ==> distance between the nearest neighbors (in meters)

    #     closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    #     # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    #     closest_points = right.loc[closest]

    #     # Ensure that the index corresponds the one in left_gdf
    #     closest_points = closest_points.reset_index(drop=True)

    #     # Add distance if requested
    #     if return_dist:
    #         # Convert to meters from radians
    #         earth_radius = 6371000  # meters
    #         closest_points['distance'] = dist * earth_radius

    #     return closest_points
    # def nearest(row, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    #     """Find the nearest point and return the corresponding value from specified column.\n
    #     ex:\n
    #     comparPts[srcCol] = comparPts.apply(nearest, df1=comparPts, df2=sumry, src_column=srcCol, axis=1)
    #     """
    #     # Find the geometry that is closest
    #     from shapely.ops import nearest_points
    #     geom_union = df2.unary_union
    #     nearestNeighbor = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    #     # Get the corresponding value from df2 (matching is based on the geometry)
    #     value = df2[nearestNeighbor][src_column].to_numpy()[0]
    #     return value
    def gdfFromFrame(df,dfXY,coordsys=None,dropXYcols=True):
        '''returns a geodataframe from a pd df and a dfXY of the same length whose columns are X,Y'''
        DF= df.copy()
        DF.index = pd.MultiIndex.from_frame(dfXY)
        DF = DF.reset_index()
        # print (gpd.GeoDataFrame(DF, geometry=gpd.points_from_xy(DF.X,DF.Y),crs = coordsys))
        gdf = gpd.GeoDataFrame(DF, geometry=gpd.points_from_xy(DF.X,DF.Y))
        if coordsys:
            gdf.crs = coordsys
        if dropXYcols:
            gdf = gdf.drop(['X','Y'],axis=1)
        return gdf
    def randPts(n=100,ret='geoseries',rangex=5000,rangey=5000,seedx=0,seedy=0,leftbound = (3020837,13595634),coors='epsg:6588'):
        '''returns Geoseries (or GDF) of pseudorandom pts\n
        great for testing, use with buffer to make polygons\n
        ret!='geoseries' will return a GDF'''
        if seedx==0: seedx = n
        if seedy==0: seedy = n*2
        np.random.seed(seedx)
        x100 = np.random.rand(n)*rangex + leftbound[0]
        np.random.seed(seedy)
        y100 = np.random.rand(n)*rangey + leftbound[1]
        if ret=='geoseries':
            pts = gpd.GeoSeries(gpd.points_from_xy(x100,y100),crs=coors)
        else:
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x100,y100),crs=coors)
        return pts
    def projekt(GDF,projectToGeom,normalized=False,how='nearest'):
        '''returns pd series\n
        shapely project should do according to docs but doesn't work with line projected to polygon\n\n
        Returns the distance along the geom geometric object to a point nearest for each feature of GDF.\n
        how in {'nearest','all', 'max','min','mean' or any other agg func}\n
        mean, min, max find all pts and then return the mean, min, or max projected station\n
        all returns all pts'''
        gdf = gp.asGDF(GDF)

        if how=='nearest':
            nearpts = gdf[g].map(lambda geom:nearest_points(projectToGeom,geom)[-1])
            nearpts

            STAs = nearpts.map(lambda pt:
                projectToGeom.project(pt,normalized=normalized))
        else:
            # all stations
            STAz = gdf[g].map(lambda gg:projectAll(gg,projectToGeom,normalized=normalized))
            STAz

            if how == 'all':
                return STAz
            
            STAs = STAz.map(lambda lst:pd.Series(lst).agg(how))

        return STAs
    def assertPolygonAreasDontOverlap(gdf):
        # gs = gdf[g] if isinstance(gdf,gpd.GeoDataFrame) else gdf
        Asum = gdf.area.sum()
        Adis = gdf.dissolve().area.iloc[0]
        assert abs(Asum-Adis)<1, f'ERROR: soil type areas of {gdf.__name__} do not add up to the area of the whole, probably overlapping polygons'        
    # def fixMultiLineString(row,snaptol=2):
    #     '''
    #     fixMultiLineStrings but for only 1 row
    #     '''
    #     raise NotImplementedError('fixMultiLineString doesnt work, shapely.ops.linemerge does it correctly.\nuse fixMultiLineStrings instead')
    #     #all this nonsense below doesn't work, shapely.ops.linemerge does it correctly
    #     strm = row
    #     try:
    #         splode = strm.explode()
    #     except Exception as e:
    #         print(row,'\n\n')
    #         raise(e)

    #     if isinstance(strm,pd.Series):
    #         if g not in splode.index[splode.index.duplicated()]:
    #             #handle whether this is being called from apply or from a 1 row gdf
    #             #one linestring inside the multilinestring, our work here is done
    #             strm[g] = splode[g]
    #             return strm
    #     elif isinstance(strm,gpd.GeoDataFrame):
    #         if len(splode.drop(g,axis=1))==1:
    #             #one linestring inside the multilinestring, our work here is done
    #             strm[g] = splode[g].iloc[0]
    #             return strm
        
    #     lns = splode[g].map(lambda geo:geo.coords)
    #     lns
    #     lns = lns.map(np.array)
    #     lns
    #     #drop Z, M, whatever
    #     lns = lns.map(lambda aray:aray[:,:2])
    #     lns
    #     lns = pd.DataFrame({'coordz':lns.to_list()})
    #     lns

    #     lns['start'],lns['end'] = lns.coordz.str[0],lns.coordz.str[-1]
    #     strt,end = np.array(lns.start.to_list()),np.array(lns.end.to_list())

    #     lns['se'],lns['es'],lns['ss'],lns['ee'] = [ xnp.KDTreee(A,B,snaptol) 
    #         for (A,B) in [(strt,end),(end,strt),(strt,strt),(end,end) ] ]
    #     lns
    #     trees = ['se','es','ss','ee']
    #     lns[trees]=  lns[trees].replace(-1,np.NaN)
        
    #     lns.index = np.arange(len(lns))

    #     #start and end segments
    #     verts = lns[lns[trees].T.count()<2]
    #     #start with 1st vert
    #     v0 = verts.index[0]
    #     v0
    #     lne = lns.loc[[v0]]
    #     lne
    #     for _ in range(len(lns)-1):
    #         #scrap current idx
    #         lns[trees]=lns[trees].replace(lne.index[-1],np.nan)
    #         lns[trees]
    #         lns[trees]=lns[trees].replace(lne.index[-1],np.nan)
    #         lastlne = lne[trees].iloc[-1]
    #         lastlne
    #         #get next idx
    #         try:
    #             i = int( lastlne.min() )#.astype(int)
    #         except:
    #             print('WARNING: Could not find 1 continuous linestring for feature:')
    #             print(strm,'\n')
    #             print(lns,'\n')
    #             print(lne.iloc[-1])
    #             return strm
    #         i
    #         if lastlne[lastlne==i].index[0] in ['ee','ss']:# the next one is flipped
    #             # print(i)
    #             # print(lns.loc[i,'start'],lns.loc[i,'end'])
    #             #flip it back
    #             # must wrap the Data Frame data args in square brackets to maintain the np.array in each cell:
    #             # print(lns.loc[i,'coordz'])
    #             lns.loc[i,'coordz'] = [ lns.loc[i,'coordz'][::-1] ]
    #             # print(lns.loc[i,'coordz'])
    #             # lns
    #             lns['start'],lns['end'] = lns.coordz.str[0],lns.coordz.str[-1]
    #             lns
    #             # print(lns.loc[i,'end'],lns.loc[i,'start'])

    #             #reparse trees
    #             strt,end = np.array(lns.start.to_list()),np.array(lns.end.to_list())
    #             lns['se'],lns['es'],lns['ss'],lns['ee'] = [ xnp.KDTreee(A,B,snaptol) 
    #                 for (A,B) in [(strt,end),(end,strt),(strt,strt),(end,end) ] ]
    #             lns[trees]=  lns[trees].replace(-1,np.NaN)

    #             #re-scrap indeces
    #             lns[trees]=lns[trees].replace({idx:np.nan for idx in lne.index.to_list()})
    #         lne = lne.append(lns.loc[[i]])
    #     lne

    #     pts = np.concatenate(lne.coordz.to_list())
    #     strm[g]= LineString(pts).simplify(0)

    #     return strm
    def fixMultiLineStrings(GDF):
        '''
        GDF: gdf of MultiLineStrings which were dissolved from separate LineStrings,\n
        The linestrings are connected end to end as a continuous line, but
        during dissolve they got screwed up, out of order, some are drawn backwards\n
        returns GDF where each feat's geo is corrected to a single, continuous LineString
        '''
        coors = GDF.crs
        GDF.loc[GDF.type=='MultiLineString',g] = GDF.loc[GDF.type=='MultiLineString',g].map(linemerge)
        assert coors == GDF.crs, f'{GDF.crs} does not match {coors}'
        return GDF

        # return GDF.apply( lambda row:gp.fixMultiLineString(row,snaptol=snaptol),
        #     axis=1)
    def checkLineOrientationUStoDS(lineGDF,dem,DStoUSinstead=False):
        '''returns a series of whether or not each line in lineGDF is oriented US to DS'''
        lineGDF = gp.fixMultiLineStrings(lineGDF)

        assert (lineGDF.type=='LineString').all()

        lns = lineGDF
        endz = lns[[g]]
        endz[g] = endz[g].map(lambda geo:LineString([geo.coords[0],geo.coords[-1]]))

        rioxda = rast.asRioxds(dem)
        rioxda

        Z = rast.sample(endz,rioxda)
        
        oriented = Z.str[0]>Z.str[-1]
        if DStoUSinstead:
            oriented = ~oriented
        oriented.name = 'DStoUS' if DStoUSinstead else 'UStoDS'
        return oriented

    def orientLinesUStoDS(lineGDF,dem,DStoUSinstead=False):
        rioxda = rast.asRioxds(dem) # just avoid reading twice
        oriented = gp.checkLineOrientationUStoDS(lineGDF,rioxda,DStoUSinstead)

        lineGDF.loc[~oriented,g] = lineGDF.loc[~oriented,g].map(reverse)

        reoriented = gp.checkLineOrientationUStoDS(lineGDF,rioxda,DStoUSinstead)
        assert reoriented.all(), reoriented.value_counts()

        return lineGDF



class rast():
    def asRioxda(rioxdaORtifpth,**kwargs):
        if isinstance(rioxdaORtifpth,PurePath):
            # with rxr.open_rasterio(tifpth) as fid:
            #     imp=fid
            return rxr.open_rasterio(rioxdaORtifpth,**kwargs)
        elif isinstance(rioxdaORtifpth,xr.DataArray):
            return rioxdaORtifpth
        else:
            print(f'WARNING:\n {rioxdaORtifpth}\n identified as type {type(rioxdaORtifpth)}, not rioxarray dataarray or TIF Path')
            # raise TypeError(f'Please supply in the form of ')
    asRioxds = asRioxda #alias
    def sample(gdf,rioxda,xydims=('x','y')):
        '''gdf or geoseries\n
        xydims correspond to rioxda x,y\n
        if gdf is Point type:\n
            returns np array of sampled Z values \n
        if LineString or Polygon:\n
            returns Series of arrays corresponding to Z values at each vertex\n
            TODO inner rings of polygons, uses just boundary\n
            TODO just returns first band of raster'''
        def _sampleDA(rioxda,gdf,xydims=('x','y')):
            '''returns new da with dim pt and coord vars x,y'''
            pts = gdf[g] if isinstance(gdf,gpd.GeoDataFrame) else gdf
                
            x,y = xydims
            xyz = rioxda.sel( {'x':xr.DataArray(pts.x, dims='pt'), 
                    'y':xr.DataArray(pts.y, dims='pt')}, 
                    method='nearest')
            xyz = xyz.isel(band=0) if 'band' in xyz.dims else xyz
            return xyz


        rioxda = rast.asRioxda(rioxda)
        gdf = gdf.to_crs(rioxda.rio.crs)
        pts = gdf[g].copy() if isinstance(gdf,gpd.GeoDataFrame) else gdf.copy()

        if pts.type.iloc[0]=='Point':
            return _sampleDA(rioxda,pts,xydims).values
        if pts.type.iloc[0]=='Polygon':
            pts = pts.boundary
        if pts.type.iloc[0]=='LineString':
            pts = pts.map(lambda line:np.array(line.coords))

        pts = pts.map(lambda pt: gpd.GeoSeries(gpd.points_from_xy(pt[:,0],pt[:,1])) )
        z = pts.map(lambda gs: _sampleDA(rioxda,gs,xydims).values)
        z.name = 'Z'
        return z
    def ZgivenXY(linestrng,rasta):
        '''returns an np array of Z vals given a linestring
        rasta is .tif Path, reads in with xarray'''
        lyne = LinetoList(linestrng)
        with rasterio.open(rasta,bigtiff='YES') as rst:
            Z = np.array([z[0] for z in rst.sample(lyne)])
        return Z
    def valcounts(rioxda,clipGDF,**kwargs):
        '''returns pd DF of value counts for region of rioxda raster within clipGDF (gpd gdf)\n
        returned DF index: 'Value', cols: ['cnt','pct'] \n
        -not tried on multiband #TODO\n
        rioxda: raster as rioxarray data array\n
        **kwargs get passed through to rioxda.rio.clip\n
        '''
        # print('rioxda',type(rioxda))
        # print(rioxda)
        # print(type(clipGDF))
        # print(clipGDF)
        
        try:
            clipper = clipGDF.copy()#['geometry'] #TODO which?
            ds_clpd = rioxda.rio.clip(clipper,**kwargs)
        except Exception as e:
            print(f'WARNING: \n{e}')
            clipper = clipGDF.copy()[['geometry']] #TODO which?
            ds_clpd = rioxda.rio.clip(clipper,**kwargs)
        
        valcounts = np.array(np.unique(ds_clpd.values, return_counts=True)).T
        valcounts = pd.DataFrame(valcounts,columns=['Value','cnt'])
        
        #drop nodata
        nodata = valcounts['Value'][~valcounts['Value'].between(1,99)]
        # assert len(nodata)==1, f'ERROR: cant find nodata value in {valcounts["Value"].unique()}'
        assert len(nodata)<2, f'ERROR too many nodata vals excluded: {nodata}'
        print('CAREFUL: rast.stats hardcoded for NLCD rasters nodata')
        valcounts = valcounts[valcounts['Value'].between(1,99)]
        valcounts['pct'] = valcounts['cnt']/valcounts['cnt'].sum()
        # print(valcounts)
        assert np.isclose(valcounts['pct'].sum(),1)
        
        return valcounts.set_index('Value')
    def setNodataAsNaN(rioxdaOrTIFpth):
        '''returns rioxda
        '''
        rioxda = rast.asRioxda(rioxdaOrTIFpth)
        if hasattr(rioxda,'_FillValue'):
            rioxda = rioxda.where(rioxda != rioxda.attrs['_FillValue'])
        rioxda.attrs['_FillValue']=np.NaN
        rioxda = rioxda.where(rioxda != np.NaN)
        return rioxda
    def cleanNodata(tif,minn,maxx):
        '''tif as rioxda or Path\n
        minn, maxx inclusive'''
        imp = rast.asRioxda(tif)
        imp = imp.where(imp>=minn).where(imp<=maxx)
        imp = rast.setNodataAsNaN(imp)
        return imp
    def checkBounds(rioxda,minn,maxx):
        assert minn<=float(rioxda.min()), f'ERROR: min {float(rioxda.min())} outside lower bound of {minn}'
        assert maxx>=float(rioxda.max()), f'ERROR: max {float(rioxda.max())} outside upper bound of {maxx}'
    # @timeit
    def clip(rioxda,clip_GDF,stripNAs=True):
        '''
        rioxda: raster as rioxarray ds\n
        clipGDF: polygon GDF (or series TODO) to clip w/\n
        output ZvarName='z'\n
        if stripNAs: subset the result to remove null values based on (band 1?)'s '_FillValue' attr
        '''
        # print('..')
        # # print(rioxda.rio.crs.wkt)
        # print('...')
        rioxda = rast.asRioxda(rioxda)
        clipGDF = gp.asGDF(clip_GDF)
        rastCRS = rioxda.rio.crs
        # rastCRS = pyproj.CRS.from_user_input(rioxda.rio.crs)
        # rastCRS = f'EPSG:{pyproj.CRS.from_wkt(rioxda.rio.crs.wkt).to_epsg()}'
        # print('there:')
        clipDissolved = clipGDF.dissolve().to_crs(rastCRS)
        # print(clipDissolved.crs)

        xybbox = np.array(clipDissolved.envelope.boundary.iloc[0].coords)
        # print(xybbox)
        
        x,y = xybbox[:,0],xybbox[:,1]
        
        # print(rioxda.rio.bounds())
        rastbnds = box(*rioxda.rio.bounds())
        assert Polygon(xybbox).overlaps(rastbnds), f'ERROR: areaGDF does not lie within bounds of rioxda raster\n\n{clipGDF}, \nPossible projection issue\nrioxdatif crs: {rastCRS}\nGDF crs: {clipGDF.crs}'
        
        # rioxda.rio.write_nodata(128, inplace=True)
        # rioxda['nodatavals'] = (128)
        # print('wrote')
        # print([min(x),min(y), max(x),max(y)])
        ds_clp = rioxda.rio.clip_box(minx=min(x), miny=min(y), maxx=max(x), maxy=max(y),auto_expand=True)
        ds_clp
        
        try:
            clipdTerr = ds_clp.rio.clip(clipDissolved[g].apply(mapping),ds_clp.rio.crs)
            #was using .to_crs(ds_clp.rio.crs??
        except:
            #if it's too big to fit in mem
            clipdTerr = ds_clp.rio.clip(clipDissolved[g].apply(mapping),ds_clp.rio.crs,from_disk=True) 

        # print(clipdTerr.attrs == rioxda.attrs)
        clipdTerr.attrs = rioxda.attrs #doesn't seem to be necessary with current xr version but may change later

        if stripNAs:
            clipdTerr=clipdTerr.where(clipdTerr != clipdTerr.attrs['_FillValue'])
        return clipdTerr
    def masquerade(E,thresh):
        return np.greater(E,thresh)*1
    def masq(a,thresh):
        return xr.apply_ufunc(rast.masquerade, a,kwargs={'thresh':thresh}, output_dtypes=np.float16,keep_attrs=True,
                            #   input_core_dims=[['x','y']]*3,output_dtypes=np.float16,output_core_dims=[['x', 'y']],
                            dask="allowed",dask_gufunc_kwargs={'allow_rechunk':False})
    def vectorize1(data_array: xr.DataArray, Zthresh,transform='from_data_array',crs='from_data_array'):#, label='Label'):
        """Return a vector representation of the input raster.
        Input
        data_array: an xarray.DataArray with boolean values (1,0) with 1 or True equal to the areas that will be turned
                    into vectors
        label: default 'Label', String, the data label that will be added to each geometry in geodataframe
        Output
        Geoseries
        NOT #  Geodataframe containing shapely geometries with data type label in a series called attribute\n
        """
        
        if transform=='from_data_array':
            transform = data_array.rio.transform()
        da = data_array
        vector = rasterio.features.shapes(
            da.data.astype('float32'),
            mask=da.data.astype('float32') == 1,  # this defines which part of array becomes polygons
            transform=transform )

        # rasterio.features.shapes outputs tuples. we only want the polygon coordinate portions of the tuples
        vectored_data = list(vector)  # put tuple output in list

        # Extract the polygon coordinates from the list
        polygons = [polygon for polygon, value in vectored_data]

        # create a list with the data label type
        Zthreshs = [Zthresh for _ in polygons]

        # Convert polygon coordinates into polygon shapes
        polygons = [shape(polygon) for polygon in polygons]
        
        if crs=='from_data_array':
            crs = pyproj.CRS.from_user_input(data_array.rio.crs)
        # Create a geopandas dataframe populated with the polygon shapes
        data_gs = gpd.GeoDataFrame(data={'Zmin': Zthreshs},
                                geometry=polygons,
                                crs=crs)

        # data_gdf = gpd.GeoDataFrame(data={'attribute': labels},
        #                            geometry=polygons,
        #                            crs=crs)
        return data_gs
    @timeit
    def vectorize(rioxda,rasterioFlipErr,ZvarName='z',contours='dynamic',zmin='fromRaster',zmax='fromRaster',smoothness=5,smoothres=2):
        '''rioxarray ds to geopandas gdf\n
        sleek, performant, and sexy\n
        rioxda: raster as rioxarray ds\n
        vectorizes to polygons at different contours, then smooths result based on the raster gridsize * smoothness\n
        rasterioFlipErr = True to correct for a rasterio bug that flips the vectorized result\n
        ZvarName: variable for raster band to vectorize on (not tested where ds has >1 var)\n

        '''
        # rastCRS = pyproj.CRS.from_wkt(rioxda.rio.crs.wkt)
        rastCRS = f'EPSG:{pyproj.CRS.from_wkt(rioxda.rio.crs.wkt).to_epsg()}'
        # = pyproj.CRS.from_user_input(rioxda.rio.crs) #in GDF-ready format
        print(rastCRS)
        gridsize = int(np.mean(np.abs(np.array(rioxda.rio.resolution()))))
        print('gridsize:',gridsize)
        minArea = (gridsize*smoothness)**2 #of polygons
        
        if contours=='dynamic':
            zrealmax = float(rioxda[ZvarName].max().values)
            if zmax=='fromRaster':
                zmax = zrealmax
            if zmin=='fromRaster':
                zmin = float(rioxda[ZvarName].min().values)
            #total contours under 50
            # band 1 |cutoff2| band 2 at step2 |cutoff3| band 3 at step3
            #1st band len 20, 2nd band len 20, 3rd band len 10
            #TODO step2,3 to be rounder #'s, like 0.1,0.25 etc
            # cutoff2,cutoff3 = 2,zmax*0.5
            # step2 = (cutoff3-cutoff2)/20
            # step3 = (zmax-cutoff3)/10
            # step2,step3 = max(0.1,step2),max(0.1,step3)
            # contours = np.concatenate([ np.array([0.05]),np.arange(.1,cutoff2,.1), np.arange(cutoff2,cutoff3,step2) , np.arange(cutoff3,zmax,step3) ])
            # contours = np.round(contours,2)
            # assert len(contours)<55

            contours = np.arange(zmin,zmax,(zmax-zmin)/500)

            contours = contours[contours< zrealmax]

        if rasterio.__version__ == '1.2.4':
            rasterioFlipErr=True
        transform = rioxda.rio.transform()
        print(transform)
        if rasterioFlipErr:
            a,e = 0,0
            b,d = 1,-1
            c,f = 0,0
            rot = rasterio.transform.Affine(a,b,c,d,e,f)
            rot

            transl = rasterio.transform.Affine(abs(transform[0]), 0.0, transform[2],0.0, abs(transform[4]), transform[5])

            afine = transl*rot

            print(f'CAREFUL: Assigning a transformation to the raster to correct for a bug in rasterio 1.2.4. If they fix the bug, change transform to default value in vectorize(). \nUsing rasterio {rasterio.__version__}')
        else:
            afine = transform

        vecs = []
        for i,zthresh in enumerate(contours):
            nowe = datetime.now()
            rmask = rast.masq(rioxda,zthresh)
            vec = rast.vectorize1(rmask[ZvarName],zthresh,afine,crs=rastCRS)
            # print(len(vec))
            vec = vec[vec.area > minArea]
            # print(len(vec))
            if vec.empty:
                print('empty')
                continue

            # nowe = datetime.now()
            # vec[g] = vec[g].buffer(0)
            # try:
            #     vec[g] = vec[g].dissolve()
            #     print(f'dissolved in { (datetime.now()-nowe) }')
            # except Exception as e:
            #     print(e)
            #     print(f'WARNING: Could not dissolve polygons for contour {zthresh}, may result in larger file size')
            

            # nowe = datetime.now()
            # print(f'smoothing: {nowe}'/)
            # vec[g]=vec[g].map(lambda p: gp.buffout_thresh(p,minArea*2,gridsize*smoothness,smoothres) )
            #UNCMT
            # print(f'smoothed in { (datetime.now()-nowe) }')
            # vec[g]=vec[g].buffer(gridsize*smoothness, join_style=1,resolution=smoothres).buffer(-gridsize*smoothness, join_style=1,resolution=smoothres)
            # print(f'halfway {(datetime.now()-nowe)}')
            # # vect=vec.copy()
            

            # print('dissolved')
            # # postpth = Path(r'C:\Users\sean.micek\Desktop\FloodMap\testpostGIS')
            # # vec.to_file(postpth/'vdissolved.gpkg',driver='GPKG')
            # # vect.to_file(postpth/'vnotdissolved.gpkg',driver='GPKG')
            # nowe = datetime.now()
            # vec[g]=vec[g].buffer(-gridsize*smoothness, join_style=1,resolution=smoothres)
            
            # vec[g] = vec[g].buffer(0) #weird 4:40, 1 min longer if it actually dissolved
            # try:
            #     vec = vec.dissolve()
            # except Exception as e:
            #     print(e)
            #     print(f'WARNING: Could not dissolve polygons for contour {zthresh}, may result in larger file size')

            vecs += [vec]
            print((datetime.now()-nowe))
            print(f'{round((i/len(contours))*100)}%')
            
        vec = gp.concat(vecs)
        assert vec.crs
        finalres = 12
        # print(vec.crs) UNCMT:
        print(f'smoothing: {nowe}')
        # vec[g]=vec[g].map(lambda p: gp.buffout(p,minArea,-gridsize*0.5) )
        vec[g]=vec[g].simplify(tolerance=gridsize*1.1,preserve_topology=False)
        print(f'smoothed in { (datetime.now()-nowe) }')
        
        # vec[g]=vec[g].buffer(-gridsize, join_style=1,resolution=finalres).buffer(gridsize, join_style=1,resolution=finalres)

        #make sure they will draw in order from low to high
        print(f'dissolving: {nowe}')
        vec = vec.dissolve(by='Zmin')
        vec = vec.sort_index() #index will be zmin
        vec = vec.reset_index()
        # vec.index = np.arange(len(vec))
        print(f'Dissolved in { (datetime.now()-nowe) }')
        # print(vec.crs)
        return vec
    def fillUSGSLiDAR(knownURLs,grupby=['b','x'],fill='y'):
        '''DL'd a patchy set of LiDAR tiles from USGS? This should help locate the missing tiles\n
        knownURLs : pd DF (not series!) of known url's in the column 'url'\n
        returns list of missing urls that should help patch it up'''
        

        knownURLs.url.to_list()[0]

        xyb =  knownURLs['url'].str.findall('x(.*)y(.*)_TX_Neches_B([0-9]+)_').str[0]
        xyb

    #     xyb.str.len().value_counts()
    #     print(xyb)
        knownURLs[['x','y','b']] = xyb.to_list()
        knownURLs

        # GDF = gpd.GeoDataFrame(geometry = knownURLs[['x','y']])
        GDF = gpd.GeoDataFrame(
            knownURLs, geometry=gpd.points_from_xy(knownURLs.x, knownURLs.y))
        GDF

        GDF[['x','y','b']] = GDF[['x','y','b']].astype(int)
        GDF

        print(GDF.b.value_counts())

        GDF.iloc[:2]

    #     get_ipython().run_line_magic('matplotlib', '')
        # base = GDF.iloc[:2].plot(color='white')
        # colrs = ['black','blue','green','yellow','red','purple']
        # for i,df in GDF.groupby('b'):
        #     df.plot(color = colrs[i],ax=base)

        GDF = GDF.sort_values(['b','x','y'])
        GDF

        grup = GDF.groupby(grupby).agg(list)
        grup

        grup['yfull'] = grup[fill].map(lambda lst: list(range(int(lst[0]),int(lst[-1])+1)))
        #replace 'y' with the missing values only:
        grup[fill] = grup.apply(lambda row: [ y for y in row['yfull'] if y not in row[fill] ],axis=1)

        needs = grup[[fill]]

        needs = needs.explode(fill)
        needs = needs.dropna()
        needs

        needs = needs.reset_index()
    #     print(needs,'\n')
    #     print(needs.apply(lambda row: f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/TX_Neches_B{row['b']}_2016/TIFF/USGS_one_meter_x{row['x']}y{row['y1']}_TX_Neches_B{row['b']}_2016.tif",1))
        
        needs['url'] = needs.apply(lambda row: f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/TX_Neches_B{row['b']}_2016/TIFF/USGS_one_meter_x{row['x']}y{row['y']}_TX_Neches_B{row['b']}_2016.tif",1)
        needs

        urlist = needs['url'].to_list()
        return urlist
    @timeit
    def TNRISgetLDRurls(regionGDF,toPth,tilekey_pth = GISdata/'TNRIStileKey'):
        '''Retrieves DL URLs for all intersecting TNRIS LiDAR DataHub tiles given a polygon shp area of interest\n
        ONLY WORKS FOR STRATMAP LiDAR sets\n
        regionGDF: geopandas GDF of region of interest\n
        tilekey_pth: a Path containing only the latest TNRIS LiDAR tile map .shp (found here: https://cdn.tnris.org/data/lidar/tnris-lidar_48_vector.zip)\n
        -uses open api.tnris.org/api/v1/collections?source_abbreviation=StratMap to get dataset IDs'''
        tileshp = list(tilekey_pth.glob('*.shp'))[-1]
        tileset = gpd.read_file( tileshp )

        # get collection id
        response = requests.get(r'https://api.tnris.org/api/v1/collections?source_abbreviation=StratMap')
        setIDs = pd.DataFrame(response.json()['results'])
        setIDs['collection_id'].duplicated().value_counts()
        # setIDs['name'].sort_values().to_list()

        regionGDF = regionGDF.to_crs(tileset.crs)
        assert regionGDF.crs==tileset.crs

        #intersect:
        tiles = tileset[tileset.tilename.str.contains('stratmap')]
        tilez = gpd.sjoin(tiles,regionGDF)
        # tilez.plot()
        tilez.to_file(toPth/'intersected.shp')

        #piece together url:
        reggie = r'(.*)_([0-9]{3,})'
        tilez['p1'] = tilez.tilename.str.findall(reggie).str[0]
        tilez['p1'],tilez['p4'] = [ tilez['p1'].str[0],tilez['p1'].str[1]]

        tilez['p2'] = tilez.apply(lambda row: row['dirname'][ row['dirname'].find(row['p1'][-3:]) +3: ],axis=1)

        setname = tilez['p2'].str[1:].str.replace('-',r')(?=.*').str.title().map(lambda st: f'(?=.*{st})')

        #can probably be optimized if too sloW:
        print('setname')
        print(setname)
        print(setname.map(lambda st: setIDs[setIDs['name'].str.contains(st)]))
        tilez.to_pickle(toPth/'tilez.pkl')
        tilez['p0'] = setname.map(lambda st: setIDs[setIDs['name'].str.contains(st)].iloc[0]['collection_id'])
        tilez['p0']=tilez['p0'].map(lambda st: f'https://data.tnris.org/{st}/resources/')

        tilez[['p3','p5']] = ['_','_dem.zip']

        ps = sorted([p for p in tilez.columns.to_list() if p.startswith('p')])
        ps = [p for p in ps if p[1:].isnumeric()]

        tilez['url']=tilez[ps].applymap(str).apply(lambda x: ''.join(x), axis=1)
        tilez.to_pickle(toPth/'tilez.pkl')
        # tilez = tilez.drop(ps,1)
        urls = list(tilez['url'].unique())
        return urls
    def zonalXY(tifpth,vecpth,outtif):
        ''''''
        tif = rxr.open_rasterio(tifpth)
        tif
        
        bldgs = gpd.read_file(vecpth)
        bldgs
        
        # bldgs.plot()
        
        # tif.plot()
        
        bldgs
        
        bldgs = bldgs.clip(box(*tif.rio.bounds()))
        # bldgs.plot()
        
        mins = []
        for i in range(len(bldgs)):
            prog(i,len(bldgs),f'Finding low points of buildings in {"shppth"}')
            clpd = rast.clip(tif,bldgs.iloc[[i]])
            minn = clpd.where(clpd==clpd.min())
            mins += [minn]
        
        
        
        mrg = rxr.merge.merge_arrays(mins)
        mrg
        
        # mrg.plot()
        
        # merge_arrays()
        
        mrg.rio.to_raster(outtif)
        print(f'Result bounced to {outtif}')

    def burnStrucsInTerrain(barrelsGDF,terrainTIF,outTIF,
                                clipvec =None,
                                spanCol='Span',
                                sampmargin=10,buffmargin=0.3,dropmargin=0):
        '''
        
        '''
        brlz = barrelsGDF
        
        if clipvec:
            clp=gp.asGDF(clipvec)
            clp=clp.to_crs(brlz.crs)
            brlz = gpd.sjoin(brlz,clp[[g]])
        
        def _buffOut(brl):
            brl[g]=brl[g].buffer(brl[spanCol]/2+buffmargin)
            return brl
        
        brlbuff = brlz.apply(_buffOut,axis=1)
        brlbuff.plot()
        
        dem = rast.asRioxda(terrainTIF)
        
        bigbuff = brlz.buffer(sampmargin)
        
        samp = rast.sample(bigbuff,dem)
        
        Z1 = samp.map(min)
        endz = rast.sample(brlz,dem)
        Z2 = endz.map(min)
        Z = pd.concat([Z1,Z2],axis=1).min(axis=1)
        
        
        brlbuff['Zmin'] = Z-dropmargin
        brlbuff
        
        burner = brlbuff.reset_index()[['Zmin',g]]
        burner
        
        assert burner.crs
        
        res = dem.rio.resolution()
        res
        
        #resolution (Union[float, Iterable[float]], optional) 
        #  A tuple of the spatial resolution of the returned data (Y, X).
        #  This includes the direction (as indicated by a positive or negative number).
        #  Typically when using most CRSs, the first number would be negative.
        res = ( -abs(res[0]) , abs(res[1]) )
        res
        
        from geocube.api.core import make_geocube
        
        out_grid = make_geocube(
            vector_data=burner,
            # measurements=["column_name"],
            resolution=res,
        )
        
        out_grid["Zmin"].rio.to_raster(outTIF)
        print(f'Structures burned into {outTIF}')
        
        return out_grid
    # def getTNRISLiDAR(regionShp,toPth):
try:
    import regionmask
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)
        print('Optional dependency for clipping ND datasets: shp.nd.clip')
class nd():
    def clip(ds,clipVec,xdim='longitude',ydim='latitude',chunkIt=False):
        '''nd dataset, must have latitude, longitude dims\n
        can be named xdim,ydim but MUST be in EPSG:4326 TODO attempt rioxarray to crs if not?\n
        if chunkIt, will chunk along time dim as 1
        '''
        print('✂️ Clip to region ✂️')

        g='geometry'
        gdf = gp.asGDF(clipVec)
        
        reg = gdf.to_crs('EPSG:4326').dissolve()[g].iloc[0]

        reg4 = regionmask.Regions([reg],name='Region',names=['region'])
        # reg4.plot(label="abbrev")

        mask=reg4.mask(ds[xdim],ds[ydim])

        clpd = ds.where(mask==0)

        # assert len(clpd.where(clpd.depth!= np.NaN)[xdim])>0, f'ERROR: data outside of clip mask {clipVec}'


        xmin,ymin,xmax,ymax = reg.bounds
        print(reg.bounds)

        wraplon = lambda lon:lon if lon>0 else lon+360

        xmin,xmax=[wraplon(x) for x in (xmin,xmax)]

        #regionmask leaves the same boundary of NaNs, perform bbox clip:
        clpd = clpd.sortby(xdim).sel( {xdim:slice(xmin,xmax)} ).sortby(ydim).sel( {ydim:slice(ymin,ymax)} )

        # try:
        #     # Still not implemented:
        #     clpd=clpd.sel(longitude=slice(xmin,xmax),latitude=slice(ymin,ymax),method='nearest')
        # except:
        #     clpd = clpd.sortby('longitude').sel(longitude=slice(xmin,xmax)).sortby('latitude').sel(latitude=slice(ymin,ymax))
        # clpd

        # clpd=clpd.dropna('longitude',how='all').dropna('latitude',how='all')
        # progress(clpd)
        if 'time' in ds.dims and chunkIt:
            clpd=clpd.chunk({'time':1,ydim:-1,xdim:-1})

        # forplot = downsamp(clpd)
        # forplot.depth.plot(robust=True)

        print(f'Clipped size decreased to {np.round(np.abs(clpd.nbytes/ds.nbytes)*100,2)}% of original')
        return clpd
    def IDW(needlesGDF,haystackDA,xname='x',yname='y',tname='time',k=3,p=1):
        '''\n
        haystackDA: data array to sample at, \n
        orthogonal multidimensional timeseries representation:\n
        dims=(station,tname), coords = (tname,xname,yname)\n
        needlesGDF: Point type GDF, geoseries or Path\n
        samples indices from inverse distance weighted aggregation using k nearest neighbors with distance to the power -p\n
        returns da with attr copied from haystackDA, dims (tname,needlesGDF.index.name), with coords (tname,needlesGDF.index.name)\n
        .to_pandas() can be applied to result
        '''
        ptsGDF = gp.asGDF(needlesGDF)
        
        if ptsGDF.type.iloc[0]!='Point':
            raise TypeError(f'needlesGDF should be Point type, instead received {ptsGDF.type.iloc[0]}. Consider using GDF.centroid or shp.gp.extractVerts(GDF) to convert to points')
        ds = haystackDA
        if not isinstance(ds,xr.DataArray):
            raise TypeError(f'haystackDA needs to be xr.DataArray, instead received {type(ds)}. Select a data var if needed before passing to IDW')
        coords = np.column_stack((ds[xname].values,ds[yname].values))
        
        btree = KDTree(coords)

        ptsgeo = ptsGDF[g] if isinstance(ptsGDF,gpd.GeoDataFrame) else ptsGDF
        BCptarray = np.column_stack([ptsgeo.x.to_numpy(),ptsgeo.y.to_numpy()])
        
        dist,c = btree.query(BCptarray,k=k) #nearest neighbor's node index
        # print('dist',dist)
        d = np.power(dist.astype(float),-1) #butt ugly but SHOULD be vectorized, human readable version: d = 1/dist
        d = np.power(d,p) #(1/dist)^p
        del dist
        
        cflat = c.flatten()
        cflat

        # loading the full thing into np because
        # https://github.com/pydata/xarray/issues/2799
        v = ds.values
        v.shape
        timeidx = v.shape.index( len(ds[tname]) )
        assert timeidx==0, 'dataset dim order different than expected'
        tsarray = v[:,c]
        tsarray.shape
        del v
        sta = ptsGDF.index.name if ptsGDF.index.name else 'pt'
        tsda = xr.DataArray(tsarray, dims=('time',sta,'k'))
        tsda
        del tsarray
        tsda['time'] = ds[tname]
        tsda.attrs = ds.attrs
        if not isinstance(ptsGDF.index,pd.RangeIndex):
            # assign coords only if incoming index actually has coords
            tsda = tsda.assign_coords({sta:(sta,ptsGDF.index)})
        tsda
        assert len(tsda[sta])==len(ptsGDF), tsda
        assert len(tsda['k'])==k,tsda

        weights = xr.DataArray(d, dims=(sta,"k"))
        assert d.shape[1]==k
        assert d.shape[0]==len(ptsGDF)
        w8ed = tsda.weighted(weights).mean(dim='k')
        return w8ed

class xzib():
    '''QGIS exhibit utils'''
    def viewportFromPoly(feat,viewWidth,aspec = 27/20,toMi=False,coordsInM=False):
        '''aspec: H/V ratio on exhibit\n
        viewWidth: width of map view along page in inches \n
        feat: shapely polygon or multipolygon to find viewport around\n
        must be projected, no deg nonsense\n
        returns shapely box of viewport to be added in geo col of atlas gpkg\n
        ex:\n
        gdf[g].map(gp.viewportFromPoly)
        gdf[g].map(lambda geo:gp.viewportFromPoly(geo,toMi=True))
        gp.viewportFromPoly(gdf.dissolve()[g][0])'''
        engscale = [1, 2, 3, 4, 5, 6]
        # vfactor = 1 #for QGIS exhibits, don't change this!
        # if toMi:
        #     vfactor = vfactor*5280
        # hfactor = vfactor*aspec
        hfactor = viewWidth
        #13.5 for QGIS exhibits, don't change this!
        if toMi:
            hfactor = hfactor*5280
        if coordsInM:
            hfactor = hfactor*0.3048
        vfactor = hfactor/aspec
        def validScale(x):
            if int(str(x)[0]) in engscale:
                if x<10:
                    return True
                else:
                    if x%10==0:
                        return True
                    else:
                        return False

        x0,y0,x1,y1 = feat.bounds
        oX,oY = [np.mean([x0,x1]),np.mean([y0,y1])]
        Hmin,Vmin = x1-x0,y1-y0

        scalemin = max(Hmin/hfactor,Vmin/vfactor)

        dig = len(str(int(scalemin)))
        incr = 10**(dig-1)
        scale = scalemin - scalemin%incr + incr
        while not validScale(scale):
            dig = len(str(int(scale)))
            incr = 10**(dig-1)
            scale += incr
        # print(scale)
        H,V = scale*hfactor,scale*vfactor
        h,v = H/2,V/2
        bbx = [(oX-h,oY-v),(oX-h,oY+v),(oX+h,oY+v),(oX+h,oY-v)]
        return Polygon(bbx)
    def typProd(atlasGDF,x,ExhibTypCol = 'ExhibTyp'):
        '''Exhibits are controlled by object x\n
        the huc8's and huc10's in hucpkg will be combined, and then multiplied by each ExhibTyp (key) in x\n
        layers and ExhibName columns will be populated by layer lists/formulas for the name\n
        columns will get populated in the order specified, so if a dynamic column depends on another specified column, it must be added after its dependency column\n
        returns the updated atlasGDF (not in place)\n
        '''
        gdfs = []
        for i,typ in enumerate(x.keys()):
            este = atlasGDF.copy()
            este[ExhibTypCol] = typ
            for col in x[typ].keys():
                if hasattr(x[typ][col], '__call__'): #is callable
                    este[col] = este.apply(x[typ][col],axis=1)
                elif isiter(x[typ][col]):
                    este[col] = '|'.join(map(str,x[typ][col]))
                else:
                    este[col] = str(x[typ][col])
            if 'id' not in este.columns:
                este['id'] = np.arange(len(este))+1
            este['id'] += 100*i*(10 if len(este)>99 else 1) #>100 and we need more
            gdfs += [este]
        gdf = pd.concat(gdfs)
        return gdf
    def bounceAtlas(gdf,pth,stem='atlas',cols='all'):
        assert gdf.crs
        assert gdf.columns[gdf.columns.str.lower().duplicated()].empty
        colz = gdf.columns.to_list() if cols=='all' else [col for col in cols if col in gdf.columns]
        assertListisin(['id','ExhibName','layers',g],colz)
        outpkg = pth/f'{stem}.gpkg'
        gdf[colz].to_file(outpkg,driver='GPKG')
        print(f'Exported atlas layer to {outpkg}')

