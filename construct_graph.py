import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.ops import split
import math
import networkx as nx
from scipy.spatial import KDTree
from tqdm import tqdm

def shortest_lines(gpd_1, gpd_2, extend=None, crs=None):
    '''
    This function solves the shortest lines from points 
    to their nearest street segments.
    
    gdf_1: geodataframe
        The shapefile of points, the coordination system should be a projected one.

    gdf_2: geodataframe
        The shapefile of street polylines, the coordination system should be a projected one.
    
    extend: float
        The extended length of shortest lines.

    crs: string
        The coordination system of output lines.
    '''
    lines = []

    for idx, row in tqdm(gpd_1.iterrows(), total=len(gpd_1), desc="Finding shortest lines"):
        # Find the nearest point on the street polylines
        nearest_point = nearest_points(row['geometry'], gpd_2.unary_union)[1]
        # Extend the shortest lines
        angle = math.atan2(nearest_point.y - row['geometry'].y, 
                           nearest_point.x - row['geometry'].x)
        x_ext = nearest_point.x + math.cos(angle)*extend
        y_ext = nearest_point.y + math.sin(angle)*extend
        line_ext = LineString([(row['geometry'].x, row['geometry'].y), (x_ext, y_ext)])

        lines.append({'geometry': line_ext})

    shortest_lines = gpd.GeoDataFrame(lines, crs=crs)
    return shortest_lines


def split_lines(splitter, gdf_2, crs=None):
    '''
    splitter: MultiLineString
        Check the splitter before using this function,
        the data type should be a MultiLineString in shapely.geometry.

    gdf_2: geodataframe
        The street polylines being split.

    crs: String
        The coordination system of output lines.
    '''

    if not isinstance(splitter, MultiLineString):
        # Convert the splitter to MultiLineString
        splitter = MultiLineString(splitter['geometry'].values)
        print("Warning: splitter is converted to a MultiLineString")

    split_lines = []

    for idx, row in tqdm(gdf_2.iterrows(), total=len(gdf_2), desc="Splitting lines"):
        split_line = row['geometry']
        result = split(split_line, splitter)
        # Iterate over the geometries in the GeometryCollection
        for geom in result.geoms:
            # Append each geometry to the list
            split_lines.append(geom)
    # Create a GeoDataFrame from the split geometries
    split_gdf = gpd.GeoDataFrame(geometry=split_lines, crs=crs)
    
    return split_gdf


def gdf_to_graph(gdf, crs=None):
    '''
    gdf: geodataframe
        The split street polylines.

    crs: String
        The coordination system of output lines.
    '''
    G = nx.Graph()
    for idx, row in gdf.iterrows():
        line = row.geometry
        start, end = line.coords[0], line.coords[-1]
        G.add_node(start, pos=(line.coords[0]))
        G.add_node(end, pos=(line.coords[-1]))
        G.add_edge(start, end, length=line.length)
    #relabel graph nodes
    new_ids = {}
    count = 0
    for node in G.nodes():
        new_ids[node] = count
        count += 1
    G = nx.relabel_nodes(G, new_ids)
    G.graph['crs'] = crs
    return G

def nearest_nodes_mapping(G, gdf_1):
    '''
    G: graph
        The converted graph from the geodataframe of split street polylines.

    gdf_1: geodataframe
        The centroid points of building polygons.
    '''
    # Extract point coordinates from networkx graph
    pos = nx.get_node_attributes(G, 'pos')
    points = np.array(list(pos.values()))
    # Extract point coordinates from building centroid points GeoDataFrame
    target_points = np.array(gdf_1['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist())
    # Build KDTree object with networkx points
    tree = KDTree(points)
    # Query tree with block points and return nearest node indices
    distances, indices = tree.query(target_points)
    # Create a dictionary mapping block IDs to nearest node indices
    nearest_nodes_map = dict(zip(gdf_1.index, indices))
    return nearest_nodes_map