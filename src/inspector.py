import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from homeharvest import scrape_property
from src.pipeline import load_or_build_trees
from src.preference_memory import register_custom_listing

def geocode_address_safe(query):
    """Attempts to geocode an address or complex name in San Antonio using osmnx."""
    try:
        import osmnx as ox
        q = str(query).strip()
        if "san antonio" not in q.lower() and "tx" not in q.lower():
            q += ", San Antonio, TX"
        point = ox.geocoder.geocode(q)
        return float(point[0]), float(point[1])
    except Exception:
        pass
    return None

def address_from_url_or_query(url_or_address):
    """
    Extracts a searchable street address, complex name, or zip from:
    - Zillow Multi-Family Community: /apartments/<city-state>/<complex-name>/<id>/
    - Zillow Single-Family / Condo: /homedetails/<slug>/<zpid>_zpid/
    - Realtor.com rentals/details: /rentals/details/<slug>
    - Realtor.com property details: /realestateandhomes-detail/<slug>
    - Raw street address strings
    """
    s = str(url_or_address).strip()
    if not s.startswith("http"):
        return s
    
    # 1. Zillow Multi-Family Community: /apartments/<city-state>/<complex-name>/...
    m_zillow_apt = re.search(r"apartments/([^/?#]+)/([^/?#]+)", s)
    if m_zillow_apt:
        city_state = m_zillow_apt.group(1).replace("-", " ")
        complex_name = m_zillow_apt.group(2).replace("-", " ")
        return f"{complex_name}, {city_state}"

    # 2. Zillow Single-Family / Condo: /homedetails/<slug>/...
    m_zillow_home = re.search(r"homedetails/([^/?#]+)", s)
    if m_zillow_home:
        slug = m_zillow_home.group(1)
        slug = re.sub(r"/[0-9]+_zpid/?$", "", slug)
        return slug.replace("-", " ")

    # 3. Realtor.com rentals/details/ pattern
    m_rentals = re.search(r"rentals/details/([^/?#]+)", s)
    if m_rentals:
        slug = m_rentals.group(1)
        slug = re.sub(r"[-_][A-Z][0-9_-]+$", "", slug)
        parts = slug.split("_")
        return " ".join(p.replace("-", " ") for p in parts)

    # 4. Realtor.com realestateandhomes-detail/ pattern
    m_realtor = re.search(r"realestateandhomes-detail/([^/?#]+)", s)
    if m_realtor:
        slug = m_realtor.group(1)
        slug = re.sub(r"_M[0-9_-]+$", "", slug)
        parts = slug.split("_")
        return " ".join(p.replace("-", " ") for p in parts)
        
    return s

def _upsert_row_to_csv(row_df, csv_path, sort_by="composite_score"):
    """Upserts a single scored listing DataFrame row into a target CSV and returns (rank, total_rows)."""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    addr = str(row_df.iloc[0].get("formatted_address", "")).strip()
    
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if "formatted_address" in existing_df.columns and addr:
                mask = existing_df["formatted_address"].astype(str).str.strip().str.lower() != addr.lower()
                existing_df = existing_df[mask]
            combined_df = pd.concat([existing_df, row_df], ignore_index=True)
        except Exception:
            combined_df = row_df.copy()
    else:
        combined_df = row_df.copy()
        
    if sort_by in combined_df.columns:
        combined_df[sort_by] = pd.to_numeric(combined_df[sort_by], errors="coerce").fillna(0.0)
        combined_df = combined_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        
    combined_df.to_csv(csv_path, index=False)
    
    rank = 1
    if "formatted_address" in combined_df.columns and addr:
        matches = combined_df.index[combined_df["formatted_address"].astype(str).str.strip().str.lower() == addr.lower()].tolist()
        if matches:
            rank = matches[0] + 1
    return rank, len(combined_df)

def inspect_and_explain_listing(
    url_or_address,
    raw_df,
    ctx,
    active_criteria,
    vision_fn=None,
    liked_urls=None,
    disliked_urls=None,
    clip_cache_path="cache/clip_photo_cache.pkl",
    cache_dir="cache",
    manual_kitchen_photo_url=None,
    manual_coords=None,
    manual_price=None,
    manual_beds=None,
    manual_baths=None,
    force_refresh_vision=False,
    auto_append_to_ledger=True,
    ledger_csv_path="outputs/san_antonio_ranked_rentals_v4.csv",
    favorites_csv_path="outputs/favorites_shortlist.csv",
    add_to_favorites=False,
    add_to_blacklist=False,
    pref_file="data/preferences.json",
    record_feedback_fn=None
):
    """
    Single-Link / Batch Address Sanity Check & Automated Ledger Injector:
    - Accepts a single URL/address OR a list of URLs/addresses.
    - Smart Multi-Floorplan Resolution: selects floorplans matching min_beds filter from apartment complexes.
    - Evaluates all 12 spatial criteria + 4-axis local CLIP vision analysis.
    - Automatically persists unlisted complexes to `preferences.json` (`tracked_custom_listings`) AND
      upserts them directly into `outputs/san_antonio_ranked_rentals_v4.csv` (and `outputs/favorites_shortlist.csv` when favored).
    """
    # Support batch list of URLs/addresses seamlessly
    if isinstance(url_or_address, (list, tuple)):
        last_df = None
        for item in url_or_address:
            if str(item).strip():
                last_df = inspect_and_explain_listing(
                    item,
                    raw_df=raw_df,
                    ctx=ctx,
                    active_criteria=active_criteria,
                    vision_fn=vision_fn,
                    liked_urls=liked_urls,
                    disliked_urls=disliked_urls,
                    clip_cache_path=clip_cache_path,
                    cache_dir=cache_dir,
                    manual_kitchen_photo_url=manual_kitchen_photo_url,
                    manual_coords=manual_coords,
                    manual_price=manual_price,
                    manual_beds=manual_beds,
                    manual_baths=manual_baths,
                    force_refresh_vision=force_refresh_vision,
                    auto_append_to_ledger=auto_append_to_ledger,
                    ledger_csv_path=ledger_csv_path,
                    favorites_csv_path=favorites_csv_path,
                    add_to_favorites=add_to_favorites,
                    add_to_blacklist=add_to_blacklist,
                    pref_file=pref_file,
                    record_feedback_fn=record_feedback_fn
                )
        return last_df

    query_str = str(url_or_address).strip()
    parsed_addr = address_from_url_or_query(query_str)
    print(f"🔍 Inspecting target listing: '{parsed_addr}'...")

    if not ctx.get('trees') or not ctx['trees'].get('grocery'):
        loc = ctx.get('config', {}).get('filters', {}).get('location', 'San Antonio, TX')
        print(f"🌲 Spatial KD-Trees not found in context. Loading/building KD-Trees for {loc}...")
        ctx['trees'] = load_or_build_trees(loc, cache_dir=cache_dir)

    min_beds_target = ctx.get('config', {}).get('filters', {}).get('min_beds', 2)
    matches = pd.DataFrame()
    is_unlisted_complex = False

    # Determine if query has a street number (e.g. "7716 Louis Pasteur Dr")
    m_num = re.search(r"\b(\d+)\b", parsed_addr)
    house_num = m_num.group(1) if m_num else None
    
    tokens = [t for t in re.split(r"\W+", parsed_addr.lower()) if len(t) >= 3 and t not in (
        "san", "antonio", "tx", "apartments", "apartment", "unit", "apt", "dr", "st", "rd", "blvd", "lane", "ln", "pkwy", "cir", "ct"
    )]
    if house_num and house_num in tokens:
        tokens.remove(house_num)

    # 1. Search in cached raw_df strictly
    if raw_df is not None and len(raw_df) > 0:
        if query_str.startswith("http") and "property_url" in raw_df.columns:
            matches = raw_df[raw_df["property_url"].astype(str).str.contains(query_str.split("?")[0], case=False, na=False)]
        
        if len(matches) == 0 and "formatted_address" in raw_df.columns:
            addrs = raw_df["formatted_address"].astype(str).str.lower()
            mask = pd.Series(True, index=raw_df.index)
            
            if house_num:
                mask = mask & addrs.str.contains(rf"\b{house_num}\b", regex=True, na=False)
            
            for tok in tokens:
                mask = mask & addrs.str.contains(rf"\b{tok}\b", regex=True, na=False)
                
            found = raw_df[mask]
            if len(found) > 0:
                target_beds = manual_beds if manual_beds is not None else min_beds_target
                unit_candidates = found[found['beds'].fillna(0) >= target_beds]
                if len(unit_candidates) > 0:
                    matches = unit_candidates.sort_values(by='list_price').head(1).copy()
                    print(f"⚡ Found {len(found)} units at complex. Selected {int(matches.iloc[0].get('beds', 2))}-Bed unit: {matches.iloc[0]['formatted_address']}")
                else:
                    with_price = found.dropna(subset=['list_price'])
                    matches = with_price.head(1).copy() if len(with_price) > 0 else found.head(1).copy()
                    print(f"⚡ Found matching building in local cache ({len(found)} units/floorplans): {matches.iloc[0]['formatted_address']}")

    # 2. Try live scrape via HomeHarvest with specific address
    if len(matches) == 0:
        print(f"🌐 Not in local scrape cache — attempting live fetch via HomeHarvest for '{parsed_addr}'...")
        try:
            live_df = scrape_property(location=parsed_addr, listing_type="for_rent", past_days=90)
            if live_df is not None and len(live_df) > 0:
                valid_live = live_df.dropna(subset=["latitude", "longitude"])
                if len(valid_live) > 0:
                    unit_candidates = valid_live[valid_live['beds'].fillna(0) >= min_beds_target]
                    matches = unit_candidates.head(1).copy() if len(unit_candidates) > 0 else valid_live.head(1).copy()
        except Exception:
            pass

    # 3. Geocoding Fallback: Resolves coordinates for exact street address or community
    if len(matches) == 0:
        is_unlisted_complex = True
        lat, lon = manual_coords if manual_coords else (None, None)
        if lat is None:
            print("🧭 Complex not indexed as active MLS unit. Resolving exact coordinates via OpenStreetMap...")
            coords = geocode_address_safe(parsed_addr)
            if coords:
                lat, lon = coords
                print(f"✅ Resolved '{parsed_addr}' to exact coordinates: ({lat:.5f}, {lon:.5f})")

        if lat is not None:
            synthetic_row = {
                "formatted_address": parsed_addr,
                "latitude": lat,
                "longitude": lon,
                "list_price": manual_price if manual_price is not None else 1650.0,
                "beds": manual_beds if manual_beds is not None else min_beds_target,
                "full_baths": manual_baths if manual_baths is not None else 2,
                "description": f"Apartment complex: {parsed_addr}. Multiple 1-bed and 2-bed floorplans available.",
                "property_url": query_str
            }
            matches = pd.DataFrame([synthetic_row])
        else:
            print(f"❌ Could not resolve coordinates for '{parsed_addr}'.")
            print("💡 Tip: Provide the exact street address with zip code, or pass `manual_coords=(lat, lon)`.")
            return None

    target_df = matches.head(1).copy().reset_index(drop=True)
    
    if manual_price is not None or pd.isna(target_df.loc[0, 'list_price']):
        target_df.loc[0, 'list_price'] = manual_price if manual_price is not None else 1650.0
    if manual_beds is not None or pd.isna(target_df.loc[0, 'beds']):
        target_df.loc[0, 'beds'] = manual_beds if manual_beds is not None else min_beds_target
    if manual_baths is not None or pd.isna(target_df.loc[0, 'full_baths']):
        target_df.loc[0, 'full_baths'] = manual_baths if manual_baths is not None else 2
    
    if manual_kitchen_photo_url:
        target_df.loc[0, 'primary_photo'] = manual_kitchen_photo_url
        print(f"📷 Injected manual kitchen photo URL for CLIP vision grading: {manual_kitchen_photo_url}")

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
            cache_path=clip_cache_path,
            force_refresh_vision=force_refresh_vision
        )
        if pd.notna(single_scored.loc[0, "vision_modern_score"]):
            single_scored.loc[0, "modern_interior_score"] = round(
                0.25 * float(single_scored.loc[0, "modern_interior_score"]) +
                0.75 * float(single_scored.loc[0, "vision_modern_score"]), 1
            )
        else:
            if pd.isna(single_scored.loc[0].get("modern_interior_score")):
                single_scored.loc[0, "modern_interior_score"] = 50.0

    r = single_scored.iloc[0]
    weights = ctx["config"]["weights"]
    comp_score = round(sum(float(r.get(k, 50.0)) * w for k, w in weights.items()), 1)
    single_scored.loc[0, "composite_score"] = comp_score
    single_scored.loc[0, "listing_status"] = "⭐ FAVORITE" if add_to_favorites else ("CUSTOM COMPLEX" if is_unlisted_complex else "INSPECTED")

    addr = str(r.get("formatted_address", parsed_addr)).strip()
    price_val = pd.to_numeric(r.get("list_price"), errors='coerce')
    beds_val = pd.to_numeric(r.get("beds"), errors='coerce')
    baths_val = pd.to_numeric(r.get("full_baths"), errors='coerce')
    
    price_disp = f"${int(price_val)}" if pd.notna(price_val) else "Range / Unlisted"
    beds_disp = f"{int(beds_val)}" if pd.notna(beds_val) else "1-2 (Mix)"
    baths_disp = f"{int(baths_val)}" if pd.notna(baths_val) else "1-2"
    
    f_cfg = ctx["config"]["filters"]
    c_cfg = ctx["config"]["cutoffs"]

    if pd.notna(price_val):
        budget_passed = (f_cfg["min_rent"] <= price_val <= f_cfg["max_rent"])
        budget_status = "✅ PASS" if budget_passed else "❌ FAIL"
    else:
        budget_passed = True
        budget_status = "ℹ️ MIXED FLOORPLANS (Set manual_price to verify exact unit)"

    if pd.notna(beds_val):
        beds_passed = (beds_val >= f_cfg["min_beds"])
        beds_status = "✅ PASS" if beds_passed else "❌ FAIL"
    else:
        beds_passed = True
        beds_status = "ℹ️ MIXED FLOORPLANS (1 & 2 Beds available in complex)"

    commute_mins = float(r.get("swri_rush_hour_mins", 99))
    commute_passed = commute_mins <= c_cfg["max_swri_commute_mins"]
    
    groc_dist = float(r.get("grocery_dist_m", 9999))
    groc_passed = groc_dist <= c_cfg["max_grocery_dist_m"]
    
    park_dist = float(r.get("park_dist_m", 9999))
    park_passed = park_dist <= c_cfg["max_park_trail_dist_m"]

    print("=" * 88)
    print(f"🏠 LISTING SANITY CHECK: {addr}")
    print(f"💵 Price: {price_disp}/mo | 🛏️ {beds_disp} Beds / {baths_disp} Baths | 🏗️ Age: {r.get('property_age_years', 'Unknown')} yrs | 🏆 Composite Score: {comp_score}/100")
    print("-" * 88)
    print("🚦 HARD CUTOFF GATEKEEPER AUDIT:")
    print(f"  • {'Rent Budget':<30}: {price_disp:<32} (Rule: ${f_cfg['min_rent']} - ${f_cfg['max_rent']}) -> {budget_status}")
    print(f"  • {'Min Bedrooms':<30}: {beds_disp} beds{'':<25} (Rule: >= {f_cfg['min_beds']} beds) -> {beds_status}")
    print(f"  • {'Appliances Gate':<30}: {r.get('appliances_status', ''):<32} (Rule: In-Unit W/D + Dishwasher) -> {'✅ PASS' if r.get('passes_appliance_gate', True) else '❌ FAIL'}")
    print(f"  • {'SwRI Rush-Hour Commute':<30}: {commute_mins} mins{'':<25} (Rule: <= {c_cfg['max_swri_commute_mins']} mins) -> {'✅ PASS' if commute_passed else '❌ FAIL'}")
    print(f"  • {'Grocery Distance (No Walmart)':<30}: {round(groc_dist)} m ({r.get('nearest_grocery')}){'':<10} (Rule: <= {c_cfg['max_grocery_dist_m']} m) -> {'✅ PASS' if groc_passed else '❌ FAIL'}")
    print(f"  • {'Park/Greenway Distance':<30}: {round(park_dist)} m ({r.get('nearest_park_trail')}){'':<10} (Rule: <= {c_cfg['max_park_trail_dist_m']} m) -> {'✅ PASS' if park_passed else '❌ FAIL'}")

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

    print("-" * 88)
    print("🎛️ HOW TO TWEAK SCORING TO GET MORE PLACES LIKE THIS:")
    strengths = [row_d["Criterion"] for row_d in rows_table if row_d["Raw Score (0-100)"] >= 80.0]
    bottlenecks = [row_d["Criterion"] for row_d in rows_table if row_d["Raw Score (0-100)"] < 55.0]
    failed_gates = []
    if not budget_passed: failed_gates.append("Rent Budget")
    if not beds_passed: failed_gates.append("Min Bedrooms")
    if not r.get('passes_appliance_gate', True): failed_gates.append("Appliances Gate")
    if not commute_passed: failed_gates.append("SwRI Rush-Hour Commute")
    if not groc_passed: failed_gates.append("Grocery Distance")
    if not park_passed: failed_gates.append("Park/Greenway Distance")
    
    if failed_gates:
        print(f"  ⚠️ Blocked by hard cutoffs: {failed_gates}.")
    if strengths:
        print(f"  📈 Signature Strengths (>=80/100): {strengths} -> Boost these weights.")
    if bottlenecks:
        print(f"  📉 Dragging Score Down (<55/100): {bottlenecks} -> Lower these weights.")

    # 4. Automated Persistence & Master Ranking Ledger Upsert
    custom_record = {
        "formatted_address": addr,
        "latitude": float(r.get("latitude")),
        "longitude": float(r.get("longitude")),
        "list_price": float(price_val) if pd.notna(price_val) else 1650.0,
        "beds": float(beds_val) if pd.notna(beds_val) else float(min_beds_target),
        "full_baths": float(baths_val) if pd.notna(baths_val) else 2.0,
        "primary_photo": str(r.get("primary_photo", "")) if pd.notna(r.get("primary_photo")) else None,
        "property_url": query_str if query_str.startswith("http") else str(r.get("property_url", "")),
        "description": str(r.get("description", f"Apartment complex at {addr}"))
    }

    if auto_append_to_ledger and not add_to_blacklist:
        register_custom_listing(custom_record, path=pref_file)
        if ledger_csv_path:
            rank, total_rows = _upsert_row_to_csv(single_scored, ledger_csv_path, sort_by="composite_score")
            print(f"📊 Auto-Synced to Master Ledger ({ledger_csv_path}): Ranked #{rank} out of {total_rows} properties! (Persisted in {pref_file})")

    if record_feedback_fn is not None and (add_to_favorites or add_to_blacklist):
        record_feedback_fn(
            address=addr,
            liked=bool(add_to_favorites and not add_to_blacklist),
            reason="Added via Single-Link Inspector",
            custom_listing=custom_record if not add_to_blacklist else None,
            path=pref_file
        )

    if add_to_favorites and not add_to_blacklist and favorites_csv_path:
        fav_rank, fav_total = _upsert_row_to_csv(single_scored, favorites_csv_path, sort_by="composite_score")
        print(f"⭐ Auto-Synced to Favorites Shortlist ({favorites_csv_path}): Ranked #{fav_rank} out of {fav_total} favorites!")

    print("=" * 88)
    return breakdown_df
