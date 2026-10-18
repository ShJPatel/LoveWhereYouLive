import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from scipy.spatial import cKDTree
from homeharvest import scrape_property
from src.criteria import ACTIVE_CRITERIA

def get_osm_tree(place, tags):
    gdf = ox.features_from_place(place, tags=tags).to_crs(epsg=32614)
    xy = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
    names = gdf['name'].fillna('Unnamed').tolist() if 'name' in gdf.columns else ['Unnamed']*len(xy)
    return cKDTree(xy), names

def run_pipeline(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loc = config['filters']['location']
    print(f"Building spatial KD-Trees for {loc}...")
    trees = {
        'grocery': get_osm_tree(loc, {'shop': ['supermarket', 'grocery', 'greengrocer']}),
        'park': get_osm_tree(loc, {'leisure': ['park', 'nature_reserve'], 'highway': ['cycleway', 'path', 'footway']}),
        'highway': get_osm_tree(loc, {'highway': ['motorway', 'trunk']}),
        'railway': get_osm_tree(loc, {'railway': 'rail'})
    }

    print("Scraping rental listings via HomeHarvest...")
    raw = scrape_property(location=loc, listing_type='for_rent', past_days=config['filters']['past_days'])
    df = raw.dropna(subset=['latitude', 'longitude', 'list_price']).copy()
    f_cfg = config['filters']
    df = df[
        (df['list_price'] >= f_cfg['min_rent']) &
        (df['list_price'] <= f_cfg['max_rent']) &
        (df['beds'].fillna(0) >= f_cfg['min_beds']) &
        (df['full_baths'].fillna(0) >= 1)
    ].copy()

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326').to_crs(epsg=32614)
    df['utm_x'], df['utm_y'] = gdf.geometry.x, gdf.geometry.y

    ctx = {'config': config, 'trees': trees}
    results = []
    w = config['weights']
    for _, row in df.iterrows():
        m = {}
        for fn in ACTIVE_CRITERIA:
            m.update(fn(row, ctx))
        m['composite_score'] = round(sum(m[k] * weight for k, weight in w.items()), 1)
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1) if results else df
    scored.to_csv("outputs/san_antonio_ranked_rentals_v3.csv", index=False)
    print("Pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()
