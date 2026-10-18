import os
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from scipy.spatial import cKDTree
from homeharvest import scrape_property
from src.criteria import ACTIVE_CRITERIA
from src.criteria.vision_grader import score_photos_with_local_clip

def get_osm_tree(place, tags):
    gdf = ox.features_from_place(place, tags=tags).to_crs(epsg=32614)
    xy = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
    names = gdf['name'].fillna('Unnamed').tolist() if 'name' in gdf.columns else ['Unnamed']*len(xy)
    return cKDTree(xy), names

def run_pipeline(config_path="config.yaml", run_local_clip=True, clip_max_candidates=25):
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
        results.append(m)

    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    c_cfg = config['cutoffs']
    filtered = scored[
        (scored['passes_appliance_gate'] == True) &
        (scored['swri_rush_hour_mins'] <= c_cfg['max_swri_commute_mins']) &
        (scored['grocery_dist_m'] <= c_cfg['max_grocery_dist_m']) &
        (scored['park_dist_m'] <= c_cfg['max_park_trail_dist_m'])
    ].sort_values(by='composite_score', ascending=False).reset_index(drop=True)

    if run_local_clip and len(filtered) > 0:
        print(f"Running Strategy 1: Local CLIP Vision Grader on top {min(clip_max_candidates, len(filtered))} candidates...")
        filtered = score_photos_with_local_clip(filtered, max_candidates=clip_max_candidates)
        # Blend NLP modern_interior_score (40%) + CLIP vision_modern_score (60%) for the top candidates
        has_vision = filtered['vision_modern_score'].notna()
        filtered.loc[has_vision, 'modern_interior_score'] = (
            0.40 * filtered.loc[has_vision, 'modern_interior_score'] +
            0.60 * filtered.loc[has_vision, 'vision_modern_score']
        ).round(1)
        # Recompute composite score and re-sort
        filtered['composite_score'] = sum(filtered[k] * weight for k, weight in w.items()).round(1)
        filtered = filtered.sort_values(by='composite_score', ascending=False).reset_index(drop=True)

    os.makedirs("outputs", exist_ok=True)
    filtered.to_csv("outputs/san_antonio_ranked_rentals_v3.csv", index=False)
    print(f"Saved {len(filtered)} top candidates to outputs/san_antonio_ranked_rentals_v3.csv")
    return filtered

if __name__ == "__main__":
    run_pipeline()
