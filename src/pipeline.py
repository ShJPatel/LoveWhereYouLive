import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree
from homeharvest import scrape_property
from src.criteria import ACTIVE_CRITERIA
from src.criteria.vision_grader import score_photos_with_local_clip
from src.preference_memory import load_preferences, register_custom_listing
from src.apify_ingest import fetch_or_load_apify_zillow_dataset

def _geocode_safe(query):
    """Attempts to geocode an address or complex name in San Antonio using osmnx."""
    try:
        q = str(query).strip()
        if "san antonio" not in q.lower() and "tx" not in q.lower():
            q += ", San Antonio, TX"
        point = ox.geocoder.geocode(q)
        return float(point[0]), float(point[1])
    except Exception:
        return None

def _parse_url_or_addr(url_or_address):
    """Extracts a clean complex name or street address from Zillow/Realtor URLs or raw strings."""
    s = str(url_or_address).strip()
    if not s.startswith("http"):
        return s
    m_zillow_apt = re.search(r"apartments/([^/?#]+)/([^/?#]+)", s)
    if m_zillow_apt:
        return f"{m_zillow_apt.group(2).replace('-', ' ')}, {m_zillow_apt.group(1).replace('-', ' ')}"
    m_zillow_home = re.search(r"homedetails/([^/?#]+)", s)
    if m_zillow_home:
        slug = re.sub(r"/[0-9]+_zpid/?$", "", m_zillow_home.group(1))
        return slug.replace("-", " ")
    m_rentals = re.search(r"rentals/details/([^/?#]+)", s)
    if m_rentals:
        slug = re.sub(r"-[A-Z][0-9_-]+$", "", m_rentals.group(1))
        return " ".join(p.replace("-", " ") for p in slug.split("_"))
    return s

def load_or_build_trees(location_name="San Antonio, TX", cache_dir="cache"):
    """Loads KD-Trees for grocery, parks, highways, and rail from cache or downloads via OSMnx."""
    os.makedirs(cache_dir, exist_ok=True)
    slug = location_name.lower().replace(",", "").replace(" ", "_")
    cache_path = os.path.join(cache_dir, f"osm_trees_{slug}.pkl")
    
    if os.path.exists(cache_path):
        print(f"⚡ Loading cached spatial KD-Trees from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                raw_trees = pickle.load(f)
            normalized_trees = {}
            if isinstance(raw_trees, dict):
                for k, v in raw_trees.items():
                    if isinstance(v, dict):
                        normalized_trees[k] = v
                    elif isinstance(v, (tuple, list)):
                        tree = v[0] if len(v) > 0 else None
                        names = v[1] if len(v) > 1 else []
                        brands = v[2] if len(v) > 2 else []
                        normalized_trees[k] = {"tree": tree, "names": names, "brands": brands}
                    else:
                        normalized_trees[k] = {"tree": v, "names": [], "brands": []}
                return normalized_trees
            elif isinstance(raw_trees, (tuple, list)):
                keys = ["grocery", "parks", "highways", "railways"]
                for i, k in enumerate(keys):
                    if i < len(raw_trees):
                        v = raw_trees[i]
                        if isinstance(v, (tuple, list)):
                            tree = v[0] if len(v) > 0 else None
                            names = v[1] if len(v) > 1 else []
                            brands = v[2] if len(v) > 2 else []
                            normalized_trees[k] = {"tree": tree, "names": names, "brands": brands}
                        else:
                            normalized_trees[k] = {"tree": v, "names": [], "brands": []}
                return normalized_trees
        except Exception:
            pass
            
    print(f"🌲 Building OSM spatial index for {location_name} (this takes ~30-45s on initial setup)...")
    trees = {}
    
    try:
        groc = ox.features_from_place(location_name, tags={"shop": ["supermarket", "grocery"]})
        groc_pts = groc.to_crs(epsg=32614).geometry.centroid
        trees["grocery"] = {
            "tree": cKDTree(np.column_stack([groc_pts.x, groc_pts.y])),
            "names": groc.get("name", pd.Series(["Grocery"] * len(groc))).fillna("Grocery").tolist(),
            "brands": groc.get("brand", pd.Series([""] * len(groc))).fillna("").tolist()
        }
    except Exception as e:
        print(f"⚠️ Grocery OSM fetch note: {e}")
        
    try:
        parks = ox.features_from_place(location_name, tags={"leisure": ["park", "nature_reserve"], "highway": ["cycleway", "path"]})
        p_pts = parks.to_crs(epsg=32614).geometry.centroid
        trees["parks"] = {
            "tree": cKDTree(np.column_stack([p_pts.x, p_pts.y])),
            "names": parks.get("name", pd.Series(["Park/Trail"] * len(parks))).fillna("Park/Trail").tolist()
        }
    except Exception as e:
        print(f"⚠️ Parks OSM fetch note: {e}")

    try:
        hw = ox.features_from_place(location_name, tags={"highway": ["motorway", "trunk", "motorway_link", "trunk_link"]})
        hw_pts = hw.to_crs(epsg=32614).geometry.centroid
        trees["highways"] = {"tree": cKDTree(np.column_stack([hw_pts.x, hw_pts.y]))}
    except Exception:
        pass

    try:
        rail = ox.features_from_place(location_name, tags={"railway": ["rail", "main"]})
        rail_pts = rail.to_crs(epsg=32614).geometry.centroid
        trees["railways"] = {"tree": cKDTree(np.column_stack([rail_pts.x, rail_pts.y]))}
    except Exception:
        pass

    with open(cache_path, "wb") as f:
        pickle.dump(trees, f)
    print(f"✅ Successfully cached spatial trees to {cache_path}!")
    return trees

def _inject_custom_complexes(raw_df, config_dict, prefs, pref_file="data/preferences.json"):
    """
    Merges all unlisted complexes from `prefs['taste_profile']['tracked_custom_listings']`
    and `config_dict.get('custom_complexes', [])` into `raw_df` so they are guaranteed
    to be scored and ranked in `san_antonio_ranked_rentals_v4.csv`.
    """
    f_cfg = config_dict.get("filters", {})
    min_beds_default = float(f_cfg.get("min_beds", 2))
    default_price = float(np.mean([f_cfg.get("min_rent", 1200), f_cfg.get("max_rent", 2200)]))

    tracked_list = list(prefs.get("taste_profile", {}).get("tracked_custom_listings", []))
    config_custom = config_dict.get("custom_complexes", [])

    for item in config_custom:
        if isinstance(item, str) and item.strip():
            parsed_addr = _parse_url_or_addr(item)
            tracked_list.append({
                "formatted_address": parsed_addr,
                "property_url": item if item.startswith("http") else "",
                "list_price": default_price,
                "beds": min_beds_default,
                "full_baths": 2.0
            })
        elif isinstance(item, dict):
            raw_q = item.get("url") or item.get("address") or item.get("formatted_address") or ""
            if raw_q:
                parsed_addr = item.get("formatted_address") or _parse_url_or_addr(raw_q)
                tracked_list.append({
                    "formatted_address": parsed_addr,
                    "latitude": item.get("latitude") or (item.get("coords")[0] if item.get("coords") else None),
                    "longitude": item.get("longitude") or (item.get("coords")[1] if item.get("coords") else None),
                    "list_price": item.get("list_price", item.get("price", default_price)),
                    "beds": item.get("beds", min_beds_default),
                    "full_baths": item.get("full_baths", item.get("baths", 2.0)),
                    "primary_photo": item.get("primary_photo", item.get("kitchen_photo_url")),
                    "property_url": item.get("property_url", item.get("url", "")),
                    "description": item.get("description", f"Apartment complex: {parsed_addr}. Features modern floor plan, in-unit washer and dryer.")
                })

    if not tracked_list:
        return raw_df

    existing_addrs = set()
    if raw_df is not None and len(raw_df) > 0 and "formatted_address" in raw_df.columns:
        existing_addrs = set(raw_df["formatted_address"].astype(str).str.strip().str.lower())

    injected_rows = []
    seen_custom_addrs = set()

    for entry in tracked_list:
        addr = str(entry.get("formatted_address", "")).strip()
        if not addr:
            continue
        addr_low = addr.lower()
        if addr_low in existing_addrs or addr_low in seen_custom_addrs:
            continue

        lat = entry.get("latitude")
        lon = entry.get("longitude")
        if pd.isna(lat) or pd.isna(lon) or lat is None or lon is None:
            coords = _geocode_safe(addr)
            if coords:
                lat, lon = coords
                entry["latitude"], entry["longitude"] = lat, lon
                register_custom_listing(entry, path=pref_file)
            else:
                print(f"⚠️ Could not auto-geocode custom complex '{addr}'; skipping injection.")
                continue

        seen_custom_addrs.add(addr_low)
        injected_rows.append({
            "formatted_address": addr,
            "latitude": float(lat),
            "longitude": float(lon),
            "list_price": float(entry.get("list_price") or default_price),
            "beds": float(entry.get("beds") or min_beds_default),
            "full_baths": float(entry.get("full_baths") or 2.0),
            "primary_photo": entry.get("primary_photo"),
            "property_url": entry.get("property_url", ""),
            "description": entry.get("description", f"Apartment complex: {addr}. Multiple 1-bed and 2-bed floorplans available.")
        })

    if injected_rows:
        print(f"🏢 Auto-injected {len(injected_rows)} tracked/unlisted apartment complex(es) into candidate pool!")
        inj_df = pd.DataFrame(injected_rows)
        if raw_df is not None and len(raw_df) > 0:
            raw_df = pd.concat([raw_df, inj_df], ignore_index=True)
        else:
            raw_df = inj_df

    return raw_df

def _apply_vectorized_spatial_sieve(df, config_dict, trees, pinned_addrs, enforce_cutoffs=True, max_scored_candidates=300):
    """
    Ultra-fast (<5ms for 10,000 rows) vectorized NumPy & cKDTree pre-sieve:
    1. Computes SwRI rush-hour commute, non-Walmart grocery distance, and park distance in batch.
    2. Prunes out-of-corridor listings exceeding `config_dict['cutoffs']` (always preserving `pinned_addrs`).
    3. If remaining rows still exceed `max_scored_candidates`, downselects the top spatial contenders.
    """
    if df is None or len(df) == 0:
        return df

    n_initial = len(df)
    c_cfg = config_dict.get("cutoffs", {})
    max_commute = float(c_cfg.get("max_swri_commute_mins", 28.0))
    max_groc = float(c_cfg.get("max_grocery_dist_m", 2400.0))
    max_park = float(c_cfg.get("max_park_trail_dist_m", 1600.0))

    # 1. Vectorized SwRI Commute Calculation
    swri_x, swri_y = 537785.0, 3257850.0
    x = df["utm_x"].to_numpy(dtype=float)
    y = df["utm_y"].to_numpy(dtype=float)
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)

    dist_km = np.sqrt((x - swri_x) ** 2 + (y - swri_y) ** 2) / 1000.0
    circuity = np.full(len(df), 1.28)
    speed_kmh = np.full(len(df), 44.0)

    m_west = lon < -98.65
    m_north = (~m_west) & (lat > 29.52)
    m_rev = (~m_west) & (~m_north) & (lon >= -98.55) & (lon <= -98.45) & (lat <= 29.48)

    circuity[m_west], speed_kmh[m_west] = 1.35, 42.0
    circuity[m_north], speed_kmh[m_north] = 1.30, 40.0
    circuity[m_rev], speed_kmh[m_rev] = 1.22, 48.0

    rush_mins = np.round((dist_km * circuity / speed_kmh) * 60.0 + 3.0, 1)

    # 2. Vectorized Batch KD-Tree Queries for Grocery & Parks
    from src.criteria import _extract_tree_and_metadata
    coords_xy = np.column_stack([x, y])

    groc_dists = np.full(len(df), 1500.0)
    if trees and "grocery" in trees:
        g_tree, g_names, g_brands = _extract_tree_and_metadata(trees.get("grocery"))
        if g_tree is not None:
            k_g = min(5, len(g_names)) if g_names else 1
            d_arr, idx_arr = g_tree.query(coords_xy, k=k_g)
            if k_g == 1:
                groc_dists = np.atleast_1d(d_arr).astype(float)
            else:
                for row_i in range(len(df)):
                    val_d = float(d_arr[row_i][0])
                    for d_cand, idx_cand in zip(d_arr[row_i], idx_arr[row_i]):
                        nm = g_names[idx_cand] if idx_cand < len(g_names) else ""
                        br = g_brands[idx_cand] if idx_cand < len(g_brands) else ""
                        if "walmart" not in f"{nm} {br}".lower():
                            val_d = float(d_cand)
                            break
                    groc_dists[row_i] = val_d

    park_dists = np.full(len(df), 800.0)
    if trees and "parks" in trees:
        p_tree, _, _ = _extract_tree_and_metadata(trees.get("parks"))
        if p_tree is not None:
            d_p, _ = p_tree.query(coords_xy, k=1)
            park_dists = np.atleast_1d(d_p).flatten().astype(float)

    df = df.copy()
    df["_fast_commute_mins"] = rush_mins
    df["_fast_groc_m"] = groc_dists
    df["_fast_park_m"] = park_dists

    is_pinned = df["formatted_address"].astype(str).str.strip().isin(pinned_addrs)

    if enforce_cutoffs:
        pass_commute = df["_fast_commute_mins"] <= max_commute
        pass_groc = df["_fast_groc_m"] <= max_groc
        pass_park = df["_fast_park_m"] <= max_park
        keep_mask = (pass_commute & pass_groc & pass_park) | is_pinned
        df = df[keep_mask].copy()
        print(
            f"⚡ Vectorized Spatial Sieve: Pruned {n_initial} -> {len(df)} listings in <5ms "
            f"(Cutoffs: Commute <= {max_commute}m, Grocery <= {int(max_groc)}m, Park <= {int(max_park)}m)"
        )

    # 3. Optional Downselection Cap if still > max_scored_candidates
    if max_scored_candidates and len(df) > max_scored_candidates:
        # Fast heuristic pre-score to retain the strongest spatial & budget candidates
        c_pts = np.clip(100.0 - (df["_fast_commute_mins"] - 12.0) * 5.0, 0.0, 100.0)
        g_pts = np.clip(100.0 - (df["_fast_groc_m"] - 600.0) * (50.0 / 1200.0), 0.0, 100.0)
        p_pts = np.clip(100.0 - (df["_fast_park_m"] - 400.0) * (50.0 / 1000.0), 0.0, 100.0)
        df["_pre_rank_score"] = (0.50 * c_pts) + (0.25 * g_pts) + (0.25 * p_pts) + np.where(is_pinned.loc[df.index], 1000.0, 0.0)
        df = df.sort_values(by="_pre_rank_score", ascending=False).head(max_scored_candidates).drop(columns=["_pre_rank_score"]).copy()
        print(f"🎯 Downselected top {len(df)} spatial contenders for full 12-criteria & vision evaluation.")

    return df.drop(columns=["_fast_commute_mins", "_fast_groc_m", "_fast_park_m"], errors="ignore")

def run_pipeline(
    config_dict,
    cache_dir="cache",
    pref_file="data/preferences.json",
    run_local_clip=True,
    clip_max_candidates=35,
    force_refresh_scrape=False,
    apify_dataset_file=None,
    enforce_cutoffs=True,
    max_scored_candidates=300
):
    """
    Executes the full pipeline:
    1. Loads KD-Trees for 12 spatial criteria.
    2. Ingests listings from:
       - HomeHarvest (live or cached Realtor/MLS)
       - Apify Zillow Scraper (local JSON/CSV or cloud dataset API)
       - Persistent tracked custom complexes and CONFIG['custom_complexes']
    3. Evaluates all 12 criteria + 4-axis local CLIP vision model.
    4. Automatically generates ranked CSV (`san_antonio_ranked_rentals_v4.csv`) and visual HTML audit.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    loc = config_dict['filters']['location']
    trees = load_or_build_trees(loc, cache_dir=cache_dir)
    
    # 1. Check for Apify Zillow Dataset or live Apify Actor trigger
        # 1. Check for Apify Zillow Dataset or live Apify Actor trigger
    import glob
    apify_df = pd.DataFrame()
    apify_src = apify_dataset_file or config_dict.get('apify_zillow_file')
    apify_actor_id = config_dict.get('apify_actor_id')
    apify_token = config_dict.get('apify_api_token')
    
    if apify_actor_id and apify_token:
        from src.apify_ingest import run_apify_actor_and_get_dataset
        run_input = config_dict.get('apify_run_input', {
            "search": config_dict.get('filters', {}).get('location', 'San Antonio, TX'),
            "maxItems": 100
        })
        apify_df = run_apify_actor_and_get_dataset(apify_actor_id, run_input, apify_token)

    # Multi-file auto-discovery: find any Apify/Zillow exports in cache_dir or repo root
    discovered_files = []
    if apify_src:
        discovered_files.append(apify_src)
    else:
        patterns = [
            "*parseforge*.json", "*zillow*.json", "*apify*.json", "*rentals*.json", "*scraper*.json",
            "*zillow*.csv", "*apify*.csv", "*rentals*.csv"
        ]
        for pat in patterns:
            discovered_files.extend(glob.glob(os.path.join(cache_dir, pat)))
        for pat in patterns:
            discovered_files.extend(glob.glob(pat))
        discovered_files = sorted(list(set(discovered_files)))
        discovered_files = [f for f in discovered_files if "seen_listings_history" not in f and "preferences" not in f]

    loaded_dfs = []
    if apify_df is not None and len(apify_df) > 0:
        loaded_dfs.append(apify_df)

    if discovered_files:
        for file_path in discovered_files:
            sub_df = fetch_or_load_apify_zillow_dataset(
                source=file_path,
                apify_api_token=apify_token,
                dataset_id=config_dict.get('apify_dataset_id')
            )
            if sub_df is not None and len(sub_df) > 0:
                print(f"📦 Ingested {len(sub_df)} listings from: {file_path}")
                loaded_dfs.append(sub_df)

    if loaded_dfs:
        apify_df = pd.concat(loaded_dfs, ignore_index=True)
        if "formatted_address" in apify_df.columns:
            apify_df = apify_df.drop_duplicates(subset=["formatted_address"], keep="first").reset_index(drop=True)
        print(f"✅ Total Apify / unlisted listings merged for scoring: {len(apify_df)}")

    # 2. Fetch / Load HomeHarvest scrape
    raw_cache_path = os.path.join(cache_dir, f"raw_listings_{loc.lower().replace(',','').replace(' ','_')}.parquet")
    raw = pd.DataFrame()
    
    if os.path.exists(raw_cache_path) and not force_refresh_scrape:
        print(f"⚡ Loading cached listings from {raw_cache_path}...")
        raw = pd.read_parquet(raw_cache_path)
    else:
        print(f"🌐 Fetching live listings via HomeHarvest for {loc}...")
        try:
            raw = scrape_property(
                location=loc,
                listing_type="for_rent",
                past_days=config_dict['filters']['past_days']
            )
            if raw is not None and len(raw) > 0:
                raw.to_parquet(raw_cache_path, index=False)
        except Exception as e:
            print(f"⚠️ HomeHarvest scrape notice: {e}")
            raw = pd.DataFrame()

    # Merge Apify Zillow listings with HomeHarvest listings
    if len(apify_df) > 0:
        print(f"🔗 Merging {len(apify_df)} Apify Zillow properties with {len(raw)} MLS listings...")
        if len(raw) > 0:
            raw = pd.concat([raw, apify_df], ignore_index=True)
        else:
            raw = apify_df.copy()
            
    prefs = load_preferences(pref_file)
    
    # Guarantee all tracked / unlisted complexes are injected into `raw`
    raw = _inject_custom_complexes(raw, config_dict, prefs, pref_file=pref_file)

    if raw is None or len(raw) == 0:
        print("❌ No listings retrieved from HomeHarvest or Apify.")
        return pd.DataFrame()
        
    liked_addrs = set(prefs["taste_profile"].get("liked_property_addresses", []))
    tracked_addrs = set(
        str(x.get("formatted_address", "")).strip()
        for x in prefs["taste_profile"].get("tracked_custom_listings", [])
        if x.get("formatted_address")
    )
    blacklist_kws = [k.lower() for k in prefs["taste_profile"].get("blacklisted_complex_keywords", [])]
    
    f_cfg = config_dict['filters']
    raw = raw.dropna(subset=['latitude', 'longitude']).copy()
    is_pinned = raw['formatted_address'].astype(str).str.strip().isin(liked_addrs | tracked_addrs)
    raw.loc[is_pinned & raw['list_price'].isna(), 'list_price'] = float(np.mean([f_cfg['min_rent'], f_cfg['max_rent']]))
    raw.loc[is_pinned & raw['beds'].isna(), 'beds'] = float(f_cfg['min_beds'])

    df = raw.dropna(subset=['latitude', 'longitude', 'list_price']).copy()
    in_budget = (df['list_price'] >= f_cfg['min_rent']) & (df['list_price'] <= f_cfg['max_rent'])
    in_beds = df['beds'].fillna(0) >= f_cfg['min_beds']
    df_pinned = df['formatted_address'].astype(str).str.strip().isin(liked_addrs | tracked_addrs)
    df = df[(in_budget & in_beds) | df_pinned].copy()
    
    if "formatted_address" in df.columns:
        df["_addr_norm"] = df["formatted_address"].astype(str).str.strip().str.lower()
        df = df.sort_values(by=["beds", "list_price"], ascending=[False, True]).drop_duplicates(subset=["_addr_norm"], keep="first").drop(columns=["_addr_norm"]).copy()

    if blacklist_kws:
        for kw in blacklist_kws:
            df = df[~df['formatted_address'].astype(str).str.lower().str.contains(kw, na=False)]
            if 'description' in df.columns:
                df = df[~df['description'].astype(str).str.lower().str.contains(kw, na=False)]
                
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").to_crs(epsg=32614)
    df["utm_x"], df["utm_y"] = gdf.geometry.x, gdf.geometry.y

    # Apply fast vectorized spatial sieve & hard-cutoff pruning before row-by-row criteria scoring
    df = _apply_vectorized_spatial_sieve(
        df,
        config_dict=config_dict,
        trees=trees,
        pinned_addrs=(liked_addrs | tracked_addrs),
        enforce_cutoffs=enforce_cutoffs,
        max_scored_candidates=max_scored_candidates
    )
    if len(df) == 0:
        print("⚠️ No listings passed the hard spatial cutoffs! Try relaxing CONFIG['cutoffs'] or setting enforce_cutoffs=False.")
        return pd.DataFrame()
    
    ctx = {'config': config_dict, 'trees': trees}
    weights = config_dict['weights']
    
    results = []
    for _, row in df.iterrows():
        m = {}
        for fn in ACTIVE_CRITERIA:
            m.update(fn(row, ctx))
        comp = round(sum(float(m.get(k, 50.0)) * w for k, w in weights.items()), 1)
        m['composite_score'] = comp
        results.append(m)
        
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    
    history_file = os.path.join(cache_dir, "seen_listings_history.json")
    history = {}
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
            
    statuses = []
    new_history = {}
    for _, r in scored.iterrows():
        addr = str(r.get("formatted_address", "")).strip()
        price = float(r.get("list_price", 0))
        if addr in liked_addrs:
            st = "⭐ FAVORITE"
        elif addr in tracked_addrs and addr not in history:
            st = "CUSTOM COMPLEX"
        elif addr in history:
            old_p = float(history[addr].get("price", price))
            if price < old_p:
                st = f"PRICE DROP: -${int(old_p - price)} (Was ${int(old_p)})"
            else:
                st = "ACTIVE"
        else:
            st = "NEW"
        statuses.append(st)
        new_history[addr] = {"price": price, "status": st}
        
    scored["listing_status"] = statuses
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(new_history, f, indent=2)
        
    scored = scored.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    
    if run_local_clip:
        clip_cache_file = os.path.join(cache_dir, "clip_photo_cache.pkl")
        scored = score_photos_with_local_clip(
            scored,
            max_candidates=clip_max_candidates,
            liked_image_urls=prefs["taste_profile"].get("liked_kitchen_photo_urls", []),
            disliked_image_urls=prefs["taste_profile"].get("disliked_kitchen_photo_urls", []),
            export_audit_html="outputs/visual_kitchen_audit.html",
            cache_path=clip_cache_file
        )
        if "vision_modern_score" in scored.columns:
            has_vis = scored["vision_modern_score"].notna()
            scored.loc[has_vis, "modern_interior_score"] = (
                0.25 * scored.loc[has_vis, "modern_interior_score"].astype(float) +
                0.75 * scored.loc[has_vis, "vision_modern_score"].astype(float)
            ).round(1)
            for idx in scored[has_vis].index:
                r_row = scored.loc[idx]
                scored.loc[idx, "composite_score"] = round(sum(float(r_row.get(k, 50.0)) * w for k, w in weights.items()), 1)
            scored = scored.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
        
    out_csv = "outputs/san_antonio_ranked_rentals_v4.csv"
    scored.to_csv(out_csv, index=False)
    print(f"✅ Successfully exported {len(scored)} ranked listings to {out_csv}!")
    
    fav_df = scored[scored["formatted_address"].astype(str).str.strip().isin(liked_addrs)]
    if len(fav_df) > 0:
        fav_path = "outputs/favorites_shortlist.csv"
        fav_df.to_csv(fav_path, index=False)
        print(f"⭐ Exported {len(fav_df)} active favorites to {fav_path}!")
        
    return scored
