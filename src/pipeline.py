import os
import pickle
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree
from homeharvest import scrape_property
from src.criteria import ACTIVE_CRITERIA
from src.criteria.vision_grader import score_photos_with_local_clip
from src. preference_memory import load_preferences, track_multi_week_changes_and_favorites

def get_osm_tree(place, tags):
    gdf = ox.features_from_place(place, tags=tags).to_crs(epsg=32614)
    xy = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
    names = gdf['name'].fillna('Unnamed').tolist() if 'name' in gdf.columns else ['Unnamed']*len(xy)
    return cKDTree(xy), names

def load_or_build_trees(loc, cache_dir="cache", force_refresh=False):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"osm_trees_{loc.replace(', ', '_').replace(' ', '_').lower()}.pkl")
    if os.path.exists(cache_file) and not force_refresh:
        print(f"⚡ Loading cached spatial KD-Trees from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    print(f"⏳ Building spatial KD-Trees for {loc} (will cache for future runs)...")
    trees = {
        'grocery': get_osm_tree(loc, {'shop': ['supermarket', 'grocery', 'greengrocer']}),
        'park': get_osm_tree(loc, {'leisure': ['park', 'nature_reserve'], 'highway': ['cycleway', 'path', 'footway']}),
        'highway': get_osm_tree(loc, {'highway': ['motorway', 'trunk']}),
        'railway': get_osm_tree(loc, {'railway': 'rail'})
    }
    with open(cache_file, "wb") as f:
        pickle.dump(trees, f)
    return trees

def run_pipeline(
    config_dict=None,
    config_path="config.yaml",
    cache_dir="cache",
    pref_file="data/preferences.json",
    run_local_clip=True,
    clip_max_candidates=25,
    force_refresh_trees=False,
    force_refresh_scrape=False
):
    if config_dict is not None:
        config = config_dict
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    loc = config['filters']['location']
    trees = load_or_build_trees(loc, cache_dir=cache_dir, force_refresh=force_refresh_trees)

    scrape_cache_file = os.path.join(cache_dir, "raw_listings_san_antonio.parquet")
    use_cached_scrape = (
        os.path.exists(scrape_cache_file)
        and not force_refresh_scrape
        and ((os.path.getmtime(scrape_cache_file)) > 0)
    )

    if use_cached_scrape:
        print(f"⚡ Loading cached rental listings from {scrape_cache_file}...")
        raw = pd.read_parquet(scrape_cache_file)
    else:
        print("⏳ Scraping active rental listings in San Antonio via HomeHarvest...")
        raw = scrape_property(location=loc, listing_type='for_rent', past_days=config['filters']['past_days'])
        try:
            raw.to_parquet(scrape_cache_file, index=False)
            print(f"✅ Cached {len(raw)} raw listings to {scrape_cache_file}")
        except Exception:
            pass

    df = raw.dropna(subset=['latitude', 'longitude', 'list_price']).copy()
    f_cfg = config['filters']
    df = df[
        (df['list_price'] >= f_cfg['min_rent']) &
        (df['list_price'] <= f_cfg['max_rent']) &
        (df['beds'].fillna(0) >= f_cfg['min_beds']) &
        (df['full_baths'].fillna(0) >= 1)
    ].copy()

    prefs = load_preferences(pref_file)
    rejected_addrs = set(prefs['taste_profile'].get('rejected_property_addresses', []))
    dealbreakers = [kw.lower() for kw in prefs['taste_profile'].get('dealbreaker_keywords', [])]
    blacklisted_complexes = [kw.lower() for kw in prefs['taste_profile'].get('blacklisted_complex_keywords', [])]

    if rejected_addrs:
        before_len = len(df)
        df = df[~df['formatted_address'].isin(rejected_addrs)].copy()
        print(f"Memory Filter: Dropped {before_len - len(df)} previously rejected properties.")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326').to_crs(epsg=32614)
    df['utm_x'], df['utm_y'] = gdf.geometry.x, gdf.geometry.y

    ctx = {'config': config, 'trees': trees}
    results = []
    w = config['weights']
    for _, row in df.iterrows():
        desc_lower = str(row.get('description', '') or '').lower()
        addr_lower = str(row.get('formatted_address', '') or '').lower()
        if any(db in desc_lower for db in dealbreakers) or any(bc in addr_lower or bc in desc_lower for bc in blacklisted_complexes):
            results.append({'passes_appliance_gate': False})
            continue
        m = {}
        for fn in ACTIVE_CRITERIA:
            m.update(fn(row, ctx))
        if m.get('passes_appliance_gate', True):
            m['composite_score'] = round(sum(m.get(k, 50.0) * w for k, w in weights.items()), 1)
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
        print(f"Running Local CLIP Vision Grader on top {min(clip_max_candidates, len(filtered))} candidates...")
        clip_cache_file = os.path.join(cache_dir, "clip_photo_cache.pkl")
        liked_urls = prefs['taste_profile'].get('liked_kitchen_photo_urls', [])
        disliked_urls = prefs['taste_profile'].get('disliked_kitchen_photo_urls', [])

        filtered = score_photos_with_local_clip(
            filtered,
            max_candidates=clip_max_candidates,
            liked_image_urls=liked_urls,
            disliked_image_urls=disliked_urls,
            export_audit_html='outputs/visual_kitchen_audit.html',
            cache_path=clip_cache_file
        )
        has_vision = filtered['vision_modern_score'].notna()
        filtered.loc[has_vision, 'modern_interior_score'] = (
            0.25 * filtered.loc[has_vision, 'modern_interior_score'] +
            0.75 * filtered.loc[has_vision, 'vision_modern_score']
        ).round(1)
        filtered['composite_score'] = sum(filtered.get(k, 50.0) * w for k, w in weights.items()).round(1)
        filtered = filtered.sort_values(by='composite_score', ascending=False).reset_index(drop=True)

    history_file = os.path.join(cache_dir, "seen_listings_history.json")
    filtered = track_multi_week_changes_and_favorites(filtered, prefs, history_path=history_file)

    os.makedirs("outputs", exist_ok=True)
    filtered.to_csv("outputs/san_antonio_ranked_rentals_v4.csv", index=False)
    print(f"✅ Saved {len(filtered)} top candidates to outputs/san_antonio_ranked_rentals_v4.csv & outputs/visual_kitchen_audit.html")
    return filtered
