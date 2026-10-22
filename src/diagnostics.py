import time
import os
import json
import timeit
import pandas as pd

def benchmark_pipeline_stages(run_pipeline_fn, config_dict, cache_dir="cache", pref_file="data/preferences.json"):
    """
    Measures and logs the exact execution time of each sub-stage in the apartment pipeline
    to determine the optimal local execution vs caching strategy.
    """
    print("⏱️ Starting Pipeline Performance Diagnostics...")
    metrics = {}

    # Stage 1: OSM KD-Trees Loading / Building
    t0 = time.time()
    from src.pipeline import load_or_build_trees
    loc = config_dict['filters']['location']
    trees = load_or_build_trees(loc, cache_dir=cache_dir)
    metrics["osm_trees_load_sec"] = round(time.time() - t0, 3)

    # Stage 2: Listing Scrape or Cache Load
    t0 = time.time()
    from homeharvest import scrape_property
    scrape_cache_file = os.path.join(cache_dir, "raw_listings_san_antonio.parquet")
    if os.path.exists(scrape_cache_file):
        raw = pd.read_parquet(scrape_cache_file)
        metrics["scrape_source"] = "parquet_cache"
    else:
        raw = scrape_property(location=loc, listing_type='for_rent', past_days=config_dict['filters']['past_days'])
        raw.to_parquet(scrape_cache_file, index=False)
        metrics["scrape_source"] = "live_homeharvest"
    metrics["listing_scrape_or_load_sec"] = round(time.time() - t0, 3)
    metrics["total_listings_scraped"] = len(raw)

    # Stage 3: Criteria Scoring Engine
    t0 = time.time()
    from src.criteria import ACTIVE_CRITERIA
    from src.preference_memory import load_preferences
    df = raw.dropna(subset=['latitude', 'longitude', 'list_price']).copy()
    f_cfg = config_dict['filters']
    df = df[(df['list_price'] >= f_cfg['min_rent']) & (df['list_price'] <= f_cfg['max_rent']) & (df['beds'].fillna(0) >= f_cfg['min_beds'])].copy()
    
    prefs = load_preferences(pref_file)
    ctx = {'config': config_dict, 'trees': trees}
    results = []
    w = config_dict['weights']
    for _, row in df.iterrows():
        m = {}
        for fn in ACTIVE_CRITERIA:
            m.update(fn(row, ctx))
        m['composite_score'] = round(sum(m.get(k, 50.0) * w for k, w in weights.items()), 1)
        results.append(m)
    metrics["criteria_scoring_sec"] = round(time.time() - t0, 3)

    # Stage 4: CLIP Vision Grader (with embedding cache)
    t0 = time.time()
    from src.criteria.vision_grader import score_photos_with_local_clip
    scored = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    clip_cache_file = os.path.join(cache_dir, "clip_photo_cache.pkl")
    scored = score_photos_with_local_clip(
        scored.head(25),
        max_candidates=25,
        liked_image_urls=prefs['taste_profile'].get('liked_kitchen_photo_urls', []),
        disliked_image_urls=prefs['taste_profile'].get('disliked_kitchen_photo_urls', []),
        export_audit_html="outputs/visual_kitchen_audit.html",
        cache_path=clip_cache_file
    )
    metrics["clip_vision_sec"] = round(time.time() - t0, 3)

    # Save diagnostic log
    log_path = os.path.join(cache_dir, "pipeline_diagnostics.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("📊 Diagnostic Results:")
    for k, v in metrics.items():
        print(f"  • {k}: {v}")
    print(f"✅ Saved benchmark metrics to {log_path}")
    return metrics
