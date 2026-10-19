import math
import re
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
    distances, indices = tree.query([row['utm_x'], row['utm_y']], k=min(5, len(names)))
    
    if not isinstance(indices, np.ndarray):
        indices = [indices]
        distances = [distances]
        
    best_idx = indices[0]
    best_dist = distances[0]
    best_name = names[best_idx]
    
    for idx, dist in zip(indices, distances):
        name = names[idx].lower()
        if 'walmart' not in name:
            best_idx = idx
            best_dist = dist
            best_name = names[idx]
            break
            
    network_dist_m = best_dist * CIRCUITY_FACTOR
    walk_mins = network_dist_m / WALK_SPEED_M_MIN
    bike_mins = network_dist_m / BIKE_SPEED_M_MIN
    score = np.clip(100 - ((network_dist_m - 400) / (2400 - 400)) * 100, 0, 100)
    return {
        'nearest_grocery': best_name,
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

def criterion_water_softener_and_filtration(row, ctx):
    desc = str(row.get('description', '') or '').lower()
    score, tags = 40.0, []
    softener_keywords = {
        'water softener': 40, 'softener loop': 35, 'softened water': 35,
        'reverse osmosis': 25, 'ro system': 25, 'filtration system': 20,
        'water filtration': 20, 'tankless water heater': 15
    }
    for kw, pts in softener_keywords.items():
        if kw in desc:
            score += pts
            tags.append(kw)
    return {
        'water_softener_tags': ', '.join(tags) if tags else 'Standard Hard Water (No Softener Listed)',
        'water_softener_score': round(float(np.clip(score, 10.0, 100.0)), 1)
    }

def criterion_pet_friendly_and_hard_flooring(row, ctx):
    desc = str(row.get('description', '') or '').lower()
    score, tags = 60.0, []
    hard_flooring_terms = ['vinyl plank', 'lvp', 'wood floor', 'hardwood', 'tile flooring', 'stained concrete', 'no carpet']
    carpet_flags = ['wall-to-wall carpet', 'carpet in bedrooms', 'new carpet']
    cat_terms = ['cats allowed', 'cat friendly', 'pets allowed', 'pet friendly', 'no pet rent']
    
    if any(term in desc for term in hard_flooring_terms):
        score += 25.0
        tags.append('Hard Flooring / LVP')
    elif any(flag in desc for flag in carpet_flags):
        score -= 20.0
        tags.append('Carpeted Bedrooms')
    if any(term in desc for term in cat_terms):
        score += 15.0
        tags.append('Pet/Cat Friendly')
        
    return {
        'pet_flooring_tags': ', '.join(tags) if tags else 'Standard Flooring',
        'pet_flooring_score': round(float(np.clip(score, 10.0, 100.0)), 1)
    }

def criterion_advanced_junk_fee_parser(row, ctx):
    desc = str(row.get('description', '') or '').lower()
    fee_patterns = r'\$(\d+)\s*(?:/mo)?\s*(?:valet|trash|waste|package|internet|tech|pest|amenity|fee)'
    matches = re.findall(fee_patterns, desc)
    extracted_fees = sum([int(m) for m in matches])
    utilities_included = any(k in desc for k in ['water included', 'trash included', 'internet included', 'free wifi', 'all bills paid'])
    
    score = 100.0 - min(extracted_fees, 50.0)
    if utilities_included:
        score += 10.0
    return {
        'extracted_monthly_fees': extracted_fees,
        'utilities_included': utilities_included,
        'effective_rent_score': round(float(np.clip(score, 30.0, 100.0)), 1)
    }

def criterion_spatial_noise_buffer(row, ctx):
    lat, lon = row['latitude'], row['longitude']
    utm_x, utm_y = row['utm_x'], row['utm_y']
    
    rail_dist_m = 9999.0
    if 'railway' in ctx['trees']:
        rail_tree, _ = ctx['trees']['railway']
        rail_dist_m, _ = rail_tree.query([utm_x, utm_y])
        
    sat_coords = (29.5338, -98.4698)
    sat_dist_km = haversine_km(lat, lon, sat_coords[0], sat_coords[1])
    
    score, notes = 90.0, []
    if rail_dist_m < 200:
        score -= 40.0
        notes.append(f'Close to Railway ({round(rail_dist_m)}m)')
    elif rail_dist_m < 500:
        score -= 15.0
        notes.append(f'Moderate Rail Proximity ({round(rail_dist_m)}m)')
    if sat_dist_km < 6.0:
        score -= 25.0
        notes.append(f'SAT Airport Approach ({round(sat_dist_km, 1)}km)')
        
    return {
        'railway_dist_m': round(rail_dist_m) if rail_dist_m < 9000 else 9999,
        'sat_airport_dist_km': round(sat_dist_km, 1),
        'noise_buffer_notes': ', '.join(notes) if notes else 'Low Noise Corridor',
        'noise_buffer_score': round(float(np.clip(score, 20.0, 100.0)), 1)
    }


def criterion_year_built_recency(row, ctx):
    """CRITERION: Favors newer construction properties (year_built) to ensure modern structural standards."""
    year_built = row.get('year_built', np.nan)
    if pd.isna(year_built):
        return {'property_age_years': 'Unknown', 'recency_score': 50.0}
    current_year = 2026
    age_years = max(0, current_year - int(year_built))
    if age_years <= 3:
        score = 100.0
    elif age_years <= 7:
        score = 85.0
    elif age_years <= 12:
        score = 70.0
    elif age_years <= 20:
        score = 50.0
    else:
        score = np.clip(100.0 - (age_years * 2.5), 15.0, 45.0)
    return {'property_age_years': age_years, 'recency_score': round(float(score), 1)}

ACTIVE_CRITERIA = [
    criterion_year_built_recency,
    criterion_required_appliances_gatekeeper,
    criterion_garage_and_bike_storage,
    criterion_costco_proximity,
    criterion_grocery_proximity,
    criterion_park_trail_proximity,
    criterion_swri_rush_hour_commute,
    criterion_pollution_and_highway_buffer,
    criterion_modern_appliances_and_open_floorplan,
    criterion_water_softener_and_filtration,
    criterion_pet_friendly_and_hard_flooring,
    criterion_advanced_junk_fee_parser,
    criterion_spatial_noise_buffer
]
