import os
import re
import numpy as np
import pandas as pd

def _extract_tree_and_metadata(item):
    """
    Polymorphically extracts (kd_tree, names, brands) from:
    - dict: {'tree': kd_tree, 'names': [...], 'brands': [...]}
    - tuple: (kd_tree, names, brands) or (kd_tree, names) or (kd_tree,)
    - raw cKDTree object directly
    """
    if item is None:
        return None, [], []
    if isinstance(item, dict):
        return item.get('tree'), item.get('names', []), item.get('brands', [])
    elif isinstance(item, (tuple, list)):
        tree = item[0] if len(item) > 0 else None
        names = item[1] if len(item) > 1 else []
        brands = item[2] if len(item) > 2 else []
        return tree, names, brands
    else:
        return item, [], []

def score_commute(row, ctx):
    """Calculates SwRI rush-hour commute time and corridor scoring."""
    swri_x, swri_y = 537785.0, 3257850.0
    x, y = row['utm_x'], row['utm_y']
    dist_km = np.sqrt((x - swri_x)**2 + (y - swri_y)**2) / 1000.0
    
    lat, lon = row['latitude'], row['longitude']
    if lon < -98.65:
        corridor = "Far West (Loop 1604 / Hwy 151)"
        circuity = 1.35
        speed_kmh = 42.0
    elif lat > 29.52:
        corridor = "North / UTSA / I-10"
        circuity = 1.30
        speed_kmh = 40.0
    elif -98.55 <= lon <= -98.45 and lat <= 29.48:
        corridor = "Reverse Commute (Central/North)"
        circuity = 1.22
        speed_kmh = 48.0
    else:
        corridor = "General Arterial / Suburb"
        circuity = 1.28
        speed_kmh = 44.0
        
    rush_mins = round((dist_km * circuity / speed_kmh) * 60.0 + 3.0, 1)
    score = round(float(np.clip(100.0 - (rush_mins - 12.0) * 5.0, 0.0, 100.0)), 1)
    return {
        "commute_score": score,
        "swri_rush_hour_mins": rush_mins,
        "commute_corridor": corridor
    }

def score_grocery(row, ctx):
    """Calculates cycling and walking proximity to quality supermarkets (H-E-B, Trader Joe's, Whole Foods)."""
    trees = ctx.get('trees', {})
    if not trees or 'grocery' not in trees:
        return {"grocery_score": 50.0, "grocery_dist_m": 2000.0, "nearest_grocery": "Unknown", "grocery_walk_mins": 25.0, "grocery_bike_mins": 8.0}
        
    x, y = row['utm_x'], row['utm_y']
    tree_obj, names, brands = _extract_tree_and_metadata(trees.get('grocery'))
    if tree_obj is None:
        return {"grocery_score": 50.0, "grocery_dist_m": 2000.0, "nearest_grocery": "Unknown", "grocery_walk_mins": 25.0, "grocery_bike_mins": 8.0}
        
    k_val = min(5, len(names)) if names else 1
    dists, idxs = tree_obj.query([[x, y]], k=k_val)
    dists = np.atleast_1d(dists).flatten()
    idxs = np.atleast_1d(idxs).flatten()
    
    valid_name = "Supermarket"
    valid_dist = float(dists[0])
    for d, idx in zip(dists, idxs):
        name = names[idx] if idx < len(names) else "Supermarket"
        brand = brands[idx] if idx < len(brands) else ""
        full = f"{name} {brand}".lower()
        if "walmart" not in full:
            valid_name = name
            valid_dist = float(d)
            break
            
    walk_mins = round((valid_dist * 1.25 / 80.0), 1)
    bike_mins = round((valid_dist * 1.20 / 250.0), 1)
    score = round(float(np.clip(100.0 - (valid_dist - 600.0) * (50.0 / 1200.0), 0.0, 100.0)), 1)
    return {
        "grocery_score": score,
        "grocery_dist_m": round(valid_dist),
        "nearest_grocery": valid_name,
        "grocery_walk_mins": walk_mins,
        "grocery_bike_mins": bike_mins
    }

def score_park_trail(row, ctx):
    """Calculates distance to nearest municipal park, greenway trail, or nature reserve."""
    trees = ctx.get('trees', {})
    if not trees or 'parks' not in trees:
        return {"park_trail_score": 50.0, "park_dist_m": 1200.0, "nearest_park_trail": "Park/Trail", "park_walk_mins": 15.0}
        
    x, y = row['utm_x'], row['utm_y']
    tree_obj, names, _ = _extract_tree_and_metadata(trees.get('parks'))
    if tree_obj is None:
        return {"park_trail_score": 50.0, "park_dist_m": 1200.0, "nearest_park_trail": "Park/Trail", "park_walk_mins": 15.0}
        
    d, idx = tree_obj.query([[x, y]], k=1)
    dist = float(np.atleast_1d(d).flatten()[0])
    idx_val = int(np.atleast_1d(idx).flatten()[0])
    name = names[idx_val] if idx_val < len(names) else "Park/Trail"
    walk_mins = round((dist * 1.25 / 80.0), 1)
    
    score = round(float(np.clip(100.0 - (dist - 400.0) * (50.0 / 1000.0), 0.0, 100.0)), 1)
    return {
        "park_trail_score": score,
        "park_dist_m": round(dist),
        "nearest_park_trail": name,
        "park_walk_mins": walk_mins
    }

def score_highway_pollution(row, ctx):
    """Evaluates acoustic and particulate buffer distance from major highways."""
    trees = ctx.get('trees', {})
    if not trees or 'highways' not in trees:
        return {"pollution_highway_score": 75.0, "highway_buffer_m": 800}
    x, y = row['utm_x'], row['utm_y']
    tree_obj, _, _ = _extract_tree_and_metadata(trees.get('highways'))
    if tree_obj is None:
        return {"pollution_highway_score": 75.0, "highway_buffer_m": 800}
    d, _ = tree_obj.query([[x, y]], k=1)
    dist = float(np.atleast_1d(d).flatten()[0])
    score = round(float(np.clip((dist - 150.0) * (100.0 / 650.0), 10.0, 100.0)), 1)
    return {"pollution_highway_score": score, "highway_buffer_m": round(dist)}

def score_garage_bike(row, ctx):
    """Scores garage availability, private carports, and secure bike storage features."""
    txt = f"{row.get('description', '')} {row.get('formatted_address', '')}".lower()
    score = 45.0
    features = []
    if any(k in txt for k in ["attached garage", "private garage", "1 car garage", "2 car garage"]):
        score = 100.0
        features.append("Attached/Private Garage")
    elif any(k in txt for k in ["detached garage", "garage parking", "carport", "covered parking"]):
        score = 80.0
        features.append("Detached Garage / Covered Parking")
    elif "bike storage" in txt or "bike racks" in txt:
        score = 65.0
        features.append("Community Bike Storage")
    else:
        features.append("Open Lot / Uncovered")
    return {"garage_bike_score": score, "garage_bike_features": ", ".join(features)}

def score_costco(row, ctx):
    """Calculates drive distance to nearest San Antonio Costco Wholesale."""
    costcos = ctx.get('config', {}).get('costco_locations', [])
    lat, lon = row['latitude'], row['longitude']
    best_mins = 99.0
    best_name = "Costco"
    for c in costcos:
        clat, clon = c['coords']
        d_km = np.sqrt(((lat - clat) * 111.0)**2 + ((lon - clon) * 96.0)**2)
        mins = round((d_km * 1.30 / 55.0) * 60.0 + 2.0, 1)
        if mins < best_mins:
            best_mins = mins
            best_name = c['name']
    score = round(float(np.clip(100.0 - (best_mins - 8.0) * 4.5, 10.0, 100.0)), 1)
    return {"costco_score": score, "costco_drive_mins": best_mins, "nearest_costco": best_name}

def score_recency_age(row, ctx):
    """Scores property age (newer build / recent renovation vs aging construction)."""
    yb = pd.to_numeric(row.get('year_built'), errors='coerce')
    curr_yr = 2026
    if pd.isna(yb) or yb < 1920:
        return {"recency_score": 50.0, "property_age_years": "Unknown", "year_built": "N/A"}
    age = max(0, curr_yr - int(yb))
    score = round(float(np.clip(100.0 - (age * 2.2), 15.0, 100.0)), 1)
    return {"recency_score": score, "property_age_years": age, "year_built": int(yb)}

def score_water_softener(row, ctx):
    """Extracts water softener presence or whole-home filtration mentions."""
    txt = str(row.get('description', '')).lower()
    if any(k in txt for k in ["water softener", "soft water", "water filtration", "reverse osmosis"]):
        return {"water_softener_score": 100.0, "water_softener_tags": "Water Softener / Filtration Mentioned"}
    return {"water_softener_score": 40.0, "water_softener_tags": "Not Mentioned"}

def score_pet_flooring(row, ctx):
    """Independent hard flooring criterion (LVP/hardwood/tile vs wall-to-wall carpet)."""
    txt = str(row.get('description', '')).lower()
    score = 60.0
    tags = []
    if any(k in txt for k in ["no carpet", "hard surface throughout", "vinyl plank throughout", "wood floors throughout", "tile throughout"]):
        score = 100.0
        tags.append("100% Hard Surface (No Carpet)")
    elif any(k in txt for k in ["vinyl plank", "hardwood", "laminate", "tile", "lvp"]):
        score = 80.0
        tags.append("Partial Hard Flooring")
    elif "carpet" in txt:
        score = 35.0
        tags.append("Carpet Mentioned")
    else:
        tags.append("Unspecified Flooring")
    return {"pet_flooring_score": score, "pet_flooring_tags": ", ".join(tags)}

def score_effective_rent(row, ctx):
    """Parses hidden monthly fees and utilities included."""
    txt = str(row.get('description', '')).lower()
    fees = 0.0
    for m in re.finditer(r"\$(\d+)\s*(?:per month|/mo|monthly)?\s*(?:admin|valet|trash|pest|amenity|tech|fee)", txt):
        fees += float(m.group(1))
    utils = []
    if "water included" in txt or "water paid" in txt: utils.append("Water")
    if "trash included" in txt: utils.append("Trash")
    if "gas included" in txt: utils.append("Gas")
    price = pd.to_numeric(row.get('list_price'), errors='coerce')
    price_val = float(price) if pd.notna(price) else 1600.0
    eff_price = price_val + fees - (len(utils) * 35.0)
    score = round(float(np.clip(100.0 - (eff_price - 1200.0) * (60.0 / 1000.0), 20.0, 100.0)), 1)
    return {
        "effective_rent_score": score,
        "extracted_monthly_fees": round(fees),
        "utilities_included": ", ".join(utils) if utils else "None Stated"
    }

def score_noise_buffer(row, ctx):
    """Scores acoustic buffer from railways and SAT airport flight paths."""
    trees = ctx.get('trees', {})
    x, y = row['utm_x'], row['utm_y']
    rail_dist = 2000.0
    if trees and 'railways' in trees:
        tree_obj, _, _ = _extract_tree_and_metadata(trees.get('railways'))
        if tree_obj is not None:
            d, _ = tree_obj.query([[x, y]], k=1)
            rail_dist = float(np.atleast_1d(d).flatten()[0])
    lat, lon = row['latitude'], row['longitude']
    sat_dist_km = round(np.sqrt(((lat - 29.5337)*111.0)**2 + ((lon - (-98.4698))*96.0)**2), 1)
    score = 90.0
    notes = []
    if rail_dist < 200.0:
        score -= 35.0
        notes.append("Close Railway (<200m)")
    elif rail_dist < 400.0:
        score -= 15.0
        notes.append("Moderate Railway Buffer")
    if sat_dist_km < 4.0:
        score -= 25.0
        notes.append("Airport Flight Corridor")
    if not notes:
        notes.append("Quiet Noise Buffer")
    return {
        "noise_buffer_score": float(np.clip(score, 10.0, 100.0)),
        "railway_dist_m": round(rail_dist),
        "sat_airport_dist_km": sat_dist_km,
        "noise_buffer_notes": ", ".join(notes)
    }

def score_modern_interior(row, ctx):
    """Text-based modern interior baseline score (blended with vision_grader)."""
    txt = str(row.get('description', '')).lower()
    score = 50.0
    tags = []
    if any(k in txt for k in ["granite", "quartz", "marble", "solid surface"]):
        score += 14.0
        tags.append("Stone Counters")
    if any(k in txt for k in ["stainless steel", "stainless appliances"]):
        score += 14.0
        tags.append("Stainless Steel")
    if any(k in txt for k in ["farmhouse sink", "apron sink", "single basin sink", "undermount sink", "workstation sink"]):
        score += 12.0
        tags.append("Farmhouse/Single Basin Sink")
    if any(k in txt for k in ["kitchen island", "center island", "prep island", "island with seating", "breakfast bar"]):
        score += 12.0
        tags.append("Kitchen Island")
    if any(k in txt for k in ["modern", "newly renovated", "renovated", "remodeled", "contemporary"]):
        score += 8.0
        tags.append("Renovated")
    # Penalize split dual sinks and dated tile in kitchen
    if any(k in txt for k in ["double sink", "divided sink", "50/50 sink", "split sink", "tile countertops", "tile kitchen counters", "ceramic tile kitchen", "tile flooring in kitchen", "12x12 tile"]):
        score -= 15.0
        tags.append("Dated Tile / Split Dual Sink")
    return {
        "modern_interior_score": float(np.clip(score, 15.0, 95.0)),
        "interior_tags": ", ".join(tags) if tags else "Standard Baseline"
    }

def criterion_required_appliances_gatekeeper(row, ctx):
    """
    Evaluates in-unit washer/dryer and dishwasher availability.
    Crucially distinguishes physical in-unit machines included from 'hookups/connections only'.
    """
    desc = str(row.get('description', '') or '').lower()
    year_built = pd.to_numeric(row.get('year_built'), errors='coerce')
    cutoffs = ctx.get('config', {}).get('cutoffs', {})
    strict = cutoffs.get('strict_appliance_text_match', False)
    require_in_unit = cutoffs.get('require_in_unit_laundry_and_dishwasher', True)

    # 1. Hard dealbreaker rejection for shared/offsite laundry
    shared_only_flags = [
        'shared laundry', 'laundromat on site', 'coin-operated', 'community laundry',
        'laundry facility', 'on-site laundry', 'no washer', 'no laundry in unit'
    ]
    if any(flag in desc for flag in shared_only_flags) and not any(k in desc for k in ['in-unit', 'in unit', 'inside unit']):
        return {
            'appliances_status': 'REJECT: Shared/Community Laundry Only',
            'passes_appliance_gate': False,
            'wd_machine_included': False,
            'wd_connections_only': False
        }

    # 2. Distinguish W/D Hookups Only vs Machines Included
    hookups_only_flags = [
        'hookups only', 'connections only', 'w/d hookup', 'w/d connection',
        'washer/dryer connection', 'washer dryer connection', 'washer / dryer hookup',
        'washer/dryer hookups', 'washer and dryer hookups', 'washer connections', 'dryer connections'
    ]
    machines_included_flags = [
        'w/d included', 'washer/dryer included', 'washer and dryer included',
        'in-unit washer/dryer', 'in-unit washer & dryer', 'in-unit washer and dryer',
        'washer/dryer in unit', 'washer and dryer in unit', 'washer / dryer in unit',
        'full-size washer and dryer', 'stackable washer/dryer included', 'in unit w/d'
    ]

    is_hookups_only = any(h in desc for h in hookups_only_flags) and not any(inc in desc for inc in machines_included_flags)
    is_machine_included = any(inc in desc for inc in machines_included_flags)

    has_dishwasher = (
        'dishwasher' in desc or 
        ('stainless' in desc and 'appliances' in desc) or 
        (pd.notna(year_built) and year_built >= 2010)
    )

    if is_machine_included:
        status = 'Verified (In-Unit Machines Included)'
        passes = True
    elif is_hookups_only:
        status = 'W/D Hookups Only (No Machines Provided)'
        passes = not strict  # In strict mode, reject hookups-only
    elif any(t in desc for t in ['washer', 'dryer', 'w/d', 'laundry room', 'utility room']):
        status = 'Likely (W/D Mentioned - Verify in Photos)'
        passes = True
    elif strict:
        status = 'REJECT: No Laundry Confirmed in Text'
        passes = False
    else:
        status = 'Unconfirmed (Verify in Tour/Photos)'
        passes = True

    if require_in_unit and not has_dishwasher and strict:
        passes = False
        status += ' | Missing Dishwasher'

    return {
        'appliances_status': status,
        'passes_appliance_gate': passes,
        'wd_machine_included': is_machine_included,
        'wd_connections_only': is_hookups_only
    }

ACTIVE_CRITERIA = [
    criterion_required_appliances_gatekeeper,
    score_commute,
    score_grocery,
    score_park_trail,
    score_highway_pollution,
    score_garage_bike,
    score_costco,
    score_recency_age,
    score_water_softener,
    score_pet_flooring,
    score_effective_rent,
    score_noise_buffer,
    score_modern_interior
]
