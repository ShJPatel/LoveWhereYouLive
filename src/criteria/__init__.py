import math
import numpy as np
import pandas as pd

CIRCUITY_FACTOR = 1.32
WALK_SPEED_M_MIN = 80.0
BIKE_SPEED_M_MIN = 250.0

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def criterion_required_appliances_gatekeeper(row, ctx):
    desc = str(row.get('description', '') or '').lower()
    year_built = row.get('year_built', np.nan)
    strict = ctx['config']['cutoffs'].get('strict_appliance_text_match', False)

    shared_only_flags = ['shared laundry', 'laundromat on site', 'no washer', 'no laundry in unit', 'coin-operated laundry']
    if any(flag in desc for flag in shared_only_flags):
        return {'appliances_status': 'REJECT: Shared/No In-Unit Laundry', 'passes_appliance_gate': False}

    in_unit_terms = ['in-unit', 'in unit', 'washer/dryer', 'washer and dryer', 'w/d included', 'w/d connections', 'laundry room', 'utility room', 'full-size washer']
    has_in_unit = any(t in desc for t in in_unit_terms)
    has_dishwasher = 'dishwasher' in desc or ('stainless' in desc and 'appliances' in desc) or (pd.notna(year_built) and year_built >= 2005)

    if has_in_unit and has_dishwasher:
        return {'appliances_status': 'Verified (In-Unit W/D + Dishwasher)', 'passes_appliance_gate': True}
    elif strict:
        return {'appliances_status': 'REJECT: Unconfirmed in Text', 'passes_appliance_gate': False}
    return {'appliances_status': 'Likely (Verify W/D in Photos)', 'passes_appliance_gate': True}

def criterion_garage_and_bike_storage(row, ctx):
    desc = str(row.get('description', '') or '').lower()
    prop_type = str(row.get('style', '') or row.get('type', '') or '').lower()
    parking_spaces = row.get('parking_garage', np.nan)

    score = 30.0
    features = []
    if pd.notna(parking_spaces) and float(parking_spaces) >= 1:
        score += 45.0
        features.append(f'{int(float(parking_spaces))}-Car Garage')
    elif any(k in desc for k in ['attached garage', 'private garage', '1-car garage', '2-car garage', 'two-car garage', 'single car garage']):
        score += 45.0
        features.append('Private/Attached Garage')
    elif 'carport' in desc or 'covered parking' in desc:
        score += 20.0
        features.append('Covered/Carport')

    if any(k in desc for k in ['bike storage', 'bicycle storage', 'bike rack', 'bike repair', 'storage unit', 'storage closet']):
        score += 25.0
        features.append('Dedicated Bike/Gear Storage')
    if 'townhome' in prop_type or 'townhouse' in prop_type or 'townhome' in desc or 'first floor' in desc:
        score += 15.0
        features.append('Townhome/Ground Access')

    return {
        'garage_bike_features': ', '.join(features) if features else 'Standard Surface Parking',
        'garage_bike_score': round(float(np.clip(score, 0.0, 100.0)), 1)
    }

def criterion_costco_proximity(row, ctx):
    lat, lon = row['latitude'], row['longitude']
    best_name, min_road_km = '', 999.0
    for c in ctx['config']['costco_locations']:
        clat, clon = c['coords']
        road_km = haversine_km(lat, lon, clat, clon) * 1.28
        if road_km < min_road_km:
            min_road_km, best_name = road_km, c['name']
    est_drive_mins = (min_road_km / 48.0) * 60.0 + 2.0
    score = np.clip(100 - ((est_drive_mins - 7.0) / (25.0 - 7.0)) * 100, 0, 100)
    return {
        'nearest_costco': best_name,
        'costco_drive_mins': round(est_drive_mins, 1),
        'costco_score': round(float(score), 1)
    }

def criterion_grocery_proximity(row, ctx):
    tree, names = ctx['trees']['grocery']
    dist_euclidean, idx = tree.query([row['utm_x'], row['utm_y']])
    network_dist_m = dist_euclidean * CIRCUITY_FACTOR
    walk_mins = network_dist_m / WALK_SPEED_M_MIN
    bike_mins = network_dist_m / BIKE_SPEED_M_MIN
    score = np.clip(100 - ((network_dist_m - 400) / (2400 - 400)) * 100, 0, 100)
    return {
        'nearest_grocery': names[idx],
        'grocery_dist_m': round(network_dist_m),
        'grocery_walk_mins': round(walk_mins, 1),
        'grocery_bike_mins': round(bike_mins, 1),
        'grocery_score': round(float(score), 1)
    }

def criterion_park_trail_proximity(row, ctx):
    tree, names = ctx['trees']['park']
    dist_euclidean, idx = tree.query([row['utm_x'], row['utm_y']])
    network_dist_m = dist_euclidean * 1.25
    walk_mins = network_dist_m / WALK_SPEED_M_MIN
    score = np.clip(100 - ((network_dist_m - 250) / (1600 - 250)) * 100, 0, 100)
    return {
        'nearest_park_trail': names[idx],
        'park_dist_m': round(network_dist_m),
        'park_walk_mins': round(walk_mins, 1),
        'park_trail_score': round(float(score), 1)
    }

def criterion_swri_rush_hour_commute(row, ctx):
    lat, lon = row['latitude'], row['longitude']
    swri_lat, swri_lon = ctx['config']['workplace']['coords']
    road_km = haversine_km(lat, lon, swri_lat, swri_lon) * 1.30
    if lon < -98.68:
        avg_speed_kmh, corridor_note = 28.0, 'West Inbound (Heavy Culebra/1604 Traffic)'
    elif lon > -98.58:
        avg_speed_kmh, corridor_note = 50.0, 'Reverse Commute (Central/North)'
    else:
        avg_speed_kmh, corridor_note = 40.0, 'Inner West / Loop 410 Local'
    est_commute_mins = (road_km / avg_speed_kmh) * 60.0 + 3.0
    score = np.clip(100 - ((est_commute_mins - 8.0) / (30.0 - 8.0)) * 100, 0, 100)
    return {
        'swri_road_km': round(road_km, 1),
        'swri_rush_hour_mins': round(est_commute_mins, 1),
        'commute_corridor': corridor_note,
        'commute_score': round(float(score), 1)
    }

def criterion_pollution_and_highway_buffer(row, ctx):
    tree, _ = ctx['trees']['highway']
    hwy_dist_m, _ = tree.query([row['utm_x'], row['utm_y']])
    if hwy_dist_m < 100:
        score = 15.0
    elif hwy_dist_m < 250:
        score = 50.0 + ((hwy_dist_m - 100) / 150.0) * 35.0
    else:
        score = np.clip(85.0 + ((hwy_dist_m - 250) / 500.0) * 15.0, 85.0, 100.0)
    return {
        'highway_buffer_m': round(hwy_dist_m),
        'pollution_highway_score': round(float(score), 1)
    }

def criterion_modern_appliances_and_open_floorplan(row, ctx):
    desc = str(row.get('description', '') or '').lower()
    year_built = row.get('year_built', np.nan)
    score, matched_tags = 50.0, []
    positive_keywords = {
        'open floor': 18, 'open-concept': 18, 'open concept': 18,
        'stainless steel': 15, 'stainless appliances': 15,
        'quartz': 12, 'granite': 10, 'kitchen island': 12, 'island kitchen': 12,
        'gas range': 12, 'gas stove': 12, 'modern': 8, 'renovated': 10
    }
    for kw, pts in positive_keywords.items():
        if kw in desc:
            score += pts
            matched_tags.append(kw)
    if pd.notna(year_built) and year_built >= 2018:
        score += 20
        matched_tags.append(f'built_{int(year_built)}')
    return {
        'interior_tags': ', '.join(matched_tags) if matched_tags else 'Standard/Unlisted',
        'modern_interior_score': round(float(np.clip(score, 10.0, 100.0)), 1)
    }

ACTIVE_CRITERIA = [
    criterion_required_appliances_gatekeeper,
    criterion_garage_and_bike_storage,
    criterion_costco_proximity,
    criterion_grocery_proximity,
    criterion_park_trail_proximity,
    criterion_swri_rush_hour_commute,
    criterion_pollution_and_highway_buffer,
    criterion_modern_appliances_and_open_floorplan
]
