import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from homeharvest import scrape_property

def address_from_url_or_query(url_or_address):
    """Extracts a searchable street address/zip from a Realtor.com, Zillow, or Redfin URL, or returns the raw address string."""
    s = str(url_or_address).strip()
    if not s.startswith("http"):
        return s
    # Realtor.com URL pattern: .../realestateandhomes-detail/123-Main-St_San-Antonio_TX_78238_M12345...
    m_realtor = re.search(r"realestateandhomes-detail/([^/?#]+)", s)
    if m_realtor:
        slug = m_realtor.group(1)
        slug = re.sub(r"_M[0-9_-]+$", "", slug)
        parts = slug.split("_")
        return " ".join(p.replace("-", " ") for p in parts)
    # Zillow URL pattern: .../homedetails/123-Main-St-San-Antonio-TX-78238/12345_zpid/
    m_zillow = re.search(r"homedetails/([^/?#]+)", s)
    if m_zillow:
        return m_zillow.group(1).replace("-", " ")
    return s

def inspect_and_explain_listing(
    url_or_address,
    raw_df,
    ctx,
    active_criteria,
    vision_fn=None,
    liked_urls=None,
    disliked_urls=None,
    clip_cache_path="cache/clip_photo_cache.pkl",
    add_to_favorites=False,
    add_to_blacklist=False,
    pref_file="data/preferences.json",
    record_feedback_fn=None
):
    """
    Single-Link / Address Sanity Check & Diagnostic Waterfall Inspector:
    1. Finds the property in `raw_df` (by URL or fuzzy address match) or scrapes it on-the-fly via HomeHarvest.
    2. Checks every Hard Cutoff Gate (Budget, Beds/Baths, In-Unit Laundry/Dishwasher, Max Commute, Max Grocery/Park Dist)
       and shows PASS / FAIL with the exact value vs threshold.
    3. Runs all 12 criteria + 5-axis CLIP Kitchen Vision and outputs a full Waterfall Table:
       [Criterion | Raw Score (0-100) | Weight | Weighted Pts | Fleet Percentile | Diagnostic Note]
    4. Computes "How to Tweak Weights to Get More Like This": identifies which criteria this property excels at
       relative to the San Antonio average and suggests exact `CONFIG['weights']` adjustments!
    """
    query_str = str(url_or_address).strip()
    parsed_addr = address_from_url_or_query(query_str)
    print(f"🔍 Inspecting target listing: '{parsed_addr}'...")

    matches = pd.DataFrame()
    if raw_df is not None and len(raw_df) > 0:
        if query_str.startswith("http") and "property_url" in raw_df.columns:
            matches = raw_df[raw_df["property_url"].astype(str).str.contains(query_str.split("?")[0], case=False, na=False)]
        if len(matches) == 0 and "formatted_address" in raw_df.columns:
            # Try matching street number + street name tokens
            tokens = [t for t in re.split(r"\W+", parsed_addr.lower()) if t and t not in ("san", "antonio", "tx", "apt", "unit")]
            if tokens:
                mask = pd.Series(True, index=raw_df.index)
                for tok in tokens[:3]:
                    mask = mask & raw_df["formatted_address"].astype(str).str.lower().str.contains(tok, na=False)
                matches = raw_df[mask]

    if len(matches) == 0:
        print(f"🌐 Not in local scrape cache — fetching live via HomeHarvest for '{parsed_addr}'...")
        try:
            live_df = scrape_property(location=parsed_addr, listing_type="for_rent", past_days=90)
            if live_df is not None and len(live_df) > 0:
                matches = live_df.dropna(subset=["latitude", "longitude"]).head(1).copy()
        except Exception as e:
            print(f"⚠️ Live scrape error: {e}")

    if len(matches) == 0:
        print("❌ Could not locate listing in cache or live scrape. Try passing the street address (e.g. '1234 Culebra Rd, San Antonio, TX').")
        return None

    target_df = matches.head(1).copy().reset_index(drop=True)
    gdf = gpd.GeoDataFrame(target_df, geometry=gpd.points_from_xy(target_df.longitude, target_df.latitude), crs="EPSG:4326").to_crs(epsg=32614)
    target_df["utm_x"], target_df["utm_y"] = gdf.geometry.x, gdf.geometry.y

    row = target_df.iloc[0]
    m = {}
    for fn in active_criteria:
        m.update(fn(row, ctx))

    single_scored = pd.concat([target_df, pd.DataFrame([m])], axis=1)

    if vision_fn is not None:
        single_scored = vision_fn(
            single_scored,
            max_candidates=1,
            liked_image_urls=liked_urls,
            disliked_image_urls=disliked_urls,
            export_audit_html=None,
            cache_path=clip_cache_path
        )
        if pd.notna(single_scored.loc[0, "vision_modern_score"]):
            single_scored.loc[0, "modern_interior_score"] = round(
                0.25 * float(single_scored.loc[0, "modern_interior_score"]) +
                0.75 * float(single_scored.loc[0, "vision_modern_score"]), 1
            )

    r = single_scored.iloc[0]
    weights = ctx["config"]["weights"]
    comp_score = round(sum(float(r.get(k, 50.0)) * w for k, w in weights.items()), 1)
    single_scored.loc[0, "composite_score"] = comp_score

    # 1. Print Header & Hard Gatekeeper Audit
    addr = r.get("formatted_address", parsed_addr)
    price = r.get("list_price", "N/A")
    beds = r.get("beds", "N/A")
    baths = r.get("full_baths", "N/A")
    f_cfg = ctx["config"]["filters"]
    c_cfg = ctx["config"]["cutoffs"]

    print("=" * 88)
    print(f"🏠 LISTING SANITY CHECK: {addr}")
    print(f"💵 Price: ${price}/mo | 🛏️ {beds} Beds / {baths} Baths | 🏗️ Age: {r.get('property_age_years', 'Unknown')} yrs | 🏆 Composite Score: {comp_score}/100")
    print("-" * 88)
    print("🚦 HARD CUTOFF GATEKEEPER AUDIT (Why was this kept or filtered out?):")
    gates = [
        ("Rent Budget", f"${price}", f"${f_cfg['min_rent']} - ${f_cfg['max_rent']}", f_cfg["min_rent"] <= float(price or 0) <= f_cfg["max_rent"]),
        ("Min Bedrooms", f"{beds} beds", f">= {f_cfg['min_beds']} beds", float(beds or 0) >= f_cfg["min_beds"]),
        ("Appliances Gate", r.get("appliances_status", ""), "In-Unit W/D + Dishwasher", bool(r.get("passes_appliance_gate", True))),
        ("SwRI Rush-Hour Commute", f"{r.get('swri_rush_hour_mins')} mins", f"<= {c_cfg['max_swri_commute_mins']} mins", float(r.get("swri_rush_hour_mins", 99)) <= c_cfg["max_swri_commute_mins"]),
        ("Grocery Distance (No Walmart)", f"{r.get('grocery_dist_m')} m ({r.get('nearest_grocery')})", f"<= {c_cfg['max_grocery_dist_m']} m", float(r.get("grocery_dist_m", 9999)) <= c_cfg["max_grocery_dist_m"]),
        ("Park/Greenway Distance", f"{r.get('park_dist_m')} m ({r.get('nearest_park_trail')})", f"<= {c_cfg['max_park_trail_dist_m']} m", float(r.get("park_dist_m", 9999)) <= c_cfg["max_park_trail_dist_m"])
    ]
    for g_name, actual, thresh, passed in gates:
        status = "✅ PASS" if passed else "❌ FAIL (FILTERED OUT BY THIS CUTOFF!)"
        print(f"  • {g_name:<30}: {actual:<32} (Rule: {thresh}) -> {status}")

    # 2. Build Criterion-by-Criterion Waterfall Table
    note_map = {
        "commute_score": f"{r.get('swri_rush_hour_mins')} mins ({r.get('commute_corridor')})",
        "grocery_score": f"{r.get('grocery_walk_mins')}m walk / {r.get('grocery_bike_mins')}m bike to {r.get('nearest_grocery')}",
        "park_trail_score": f"{r.get('park_walk_mins')}m walk to {r.get('nearest_park_trail')}",
        "modern_interior_score": f"Vision: {r.get('vision_notes', 'N/A')} | Text: {r.get('interior_tags', '')}",
        "pollution_highway_score": f"{r.get('highway_buffer_m')}m from nearest highway",
        "garage_bike_score": f"{r.get('garage_bike_features')}",
        "costco_score": f"{r.get('costco_drive_mins')}m drive to {r.get('nearest_costco')}",
        "recency_score": f"Age: {r.get('property_age_years')} yrs (Built {r.get('year_built', 'N/A')})",
        "water_softener_score": f"{r.get('water_softener_tags')}",
        "pet_flooring_score": f"{r.get('pet_flooring_tags')}",
        "effective_rent_score": f"Extracted Fees: ${r.get('extracted_monthly_fees', 0)}/mo | Utils Incl: {r.get('utilities_included')}",
        "noise_buffer_score": f"Rail: {r.get('railway_dist_m')}m | SAT: {r.get('sat_airport_dist_km')}km ({r.get('noise_buffer_notes')})"
    }

    rows_table = []
    for k, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        val = float(r.get(k, 50.0))
        pts = round(val * w, 2)
        rows_table.append({
            "Criterion": k,
            "Raw Score (0-100)": val,
            "Weight": w,
            "Weighted Pts": pts,
            "Details / Why It Scored This": note_map.get(k, "")
        })
    breakdown_df = pd.DataFrame(rows_table)

    # 3. Suggest Weight Tweaks if the user LOVES this listing and wants more like it
    print("-" * 88)
    print("🎛️ HOW TO TWEAK SCORING TO GET MORE PLACES LIKE THIS:")
    strengths = [row_d["Criterion"] for row_d in rows_table if row_d["Raw Score (0-100)"] >= 80.0]
    bottlenecks = [row_d["Criterion"] for row_d in rows_table if row_d["Raw Score (0-100)"] < 55.0]
    failed_gates = [g[0] for g in gates if not g[3]]
    if failed_gates:
        print(f"  ⚠️ First: This listing was blocked by hard cutoffs: {failed_gates}. Relax those in `CONFIG['cutoffs']` if you want it included!")
    if strengths:
        print(f"  📈 Signature Strengths (>=80/100): {strengths} -> Boost these weights in `CONFIG['weights']` to surface similar properties.")
    if bottlenecks:
        print(f"  📉 Dragging Score Down (<55/100): {bottlenecks} -> Lower these weights if you don't care about them for this style of home.")
    print("=" * 88)

    if record_feedback_fn is not None and (add_to_favorites or add_to_blacklist):
        record_feedback_fn(
            address=addr,
            liked=bool(add_to_favorites and not add_to_blacklist),
            reason="Added via Single-Link Inspector",
            path=pref_file
        )

    return breakdown_df
