import os
import json
import re
import requests
import numpy as np
import pandas as pd

def _geocode_safe_fallback(query):
    """Geocodes an address or complex name in San Antonio using osmnx or Nominatim."""
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

def normalize_apify_zillow_dataframe(df):
    """
    Universal normalizer for Apify Zillow actors:
    - parseforge/zillow-rentals-scraper
    - maxcopell/zillow-scraper
    - themineworks/zillow-rental-listings
    - epctex/zillow-scraper
    - dtrungtin/zillow-scraper
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    df = df.copy()
    
    # 1. Unpack nested address dicts: `address` or `listingAddress`
    for addr_col in ["listingAddress", "address"]:
        if addr_col in df.columns:
            first_valid = df[addr_col].dropna().iloc[0] if len(df[addr_col].dropna()) > 0 else None
            if isinstance(first_valid, dict):
                addr_df = pd.json_normalize(df[addr_col])
                for c in addr_df.columns:
                    target_k = f"addr_{c}"
                    if target_k not in df.columns:
                        df[target_k] = addr_df[c]
                        
    def coalesce_cols(cols, default=None):
        res = None
        for c in cols:
            if c in df.columns:
                if res is None:
                    res = df[c].copy()
                else:
                    res = res.fillna(df[c])
        if res is None:
            return pd.Series([default] * len(df))
        return res
        
    street = coalesce_cols(["streetAddress", "street_address", "addr_street", "addr_streetAddress", "street", "streetName"])
    city = coalesce_cols(["city", "addr_city"], default="San Antonio")
    state = coalesce_cols(["state", "addr_state"], default="TX")
    zipcode = coalesce_cols(["zipcode", "zipCode", "addr_zipCode", "addr_zipcode", "postalCode"], default="")
    
    raw_addr = coalesce_cols(["formatted_address", "addr_full", "address", "addr_address", "unparsedAddress"])
    synth_addr = street.astype(str) + ", " + city.astype(str) + ", " + state.astype(str)
    if not zipcode.isna().all() and (zipcode != "").any():
        synth_addr = synth_addr + " " + zipcode.astype(str)
    formatted_address = raw_addr.fillna(synth_addr)
    
    list_price = coalesce_cols(["list_price", "price", "unformattedPrice", "rentZestimate", "rate", "monthlyRent"])
    beds = coalesce_cols(["beds", "bedrooms", "bedroom", "minBeds"])
    full_baths = coalesce_cols(["full_baths", "bathrooms", "bathroomsFull", "baths", "bathroom"])
    latitude = coalesce_cols(["latitude", "lat", "addr_latitude"])
    longitude = coalesce_cols(["longitude", "lon", "lng", "addr_longitude"])
    
    # Photos: Support mainImage, imageUrl, imgSrc, desktopWebHdpImageLink, hiResImageLink, etc.
    primary_photo = coalesce_cols(["primary_photo", "mainImage", "imageUrl", "imgSrc", "desktopWebHdpImageLink", "hiResImageLink", "image", "photo"])
    
    # Extract alt_photos from listingPhotos, imageUrls, photos
    alt_photo_col = None
    for p_candidate in ["listingPhotos", "imageUrls", "photos", "photoUrls", "images"]:
        if p_candidate in df.columns:
            def extract_photo_list(p_val):
                if isinstance(p_val, list):
                    clean_urls = []
                    for item in p_val:
                        if isinstance(item, dict):
                            u = item.get("url") or item.get("mixedSources", {}).get("jpeg", [{}])[-1].get("url")
                            if u: clean_urls.append(str(u))
                        elif isinstance(item, str) and item.startswith("http"):
                            clean_urls.append(item)
                    return ", ".join(clean_urls[:14])
                return ""
            alt_photo_col = df[p_candidate].apply(extract_photo_list)
            break

    if alt_photo_col is not None:
        first_from_alt = alt_photo_col.apply(lambda x: x.split(", ")[0] if x else None)
        primary_photo = primary_photo.fillna(first_from_alt)
        
    property_url = coalesce_cols(["property_url", "propertyUrl", "url", "detailUrl", "hdpUrl", "webUrl"])
    building_name = coalesce_cols(["buildingName", "complexName", "communityName"])
    
    raw_desc = coalesce_cols(["description", "homeDescription", "overview", "remarks"])
    desc_series = []
    for idx in range(len(df)):
        d = str(raw_desc.iloc[idx]).strip() if pd.notna(raw_desc.iloc[idx]) else ""
        b_name = str(building_name.iloc[idx]).strip() if pd.notna(building_name.iloc[idx]) else ""
        addr = str(formatted_address.iloc[idx]).strip()
        if not d or d.lower() == "nan" or len(d) < 5:
            d = f"{b_name} luxury apartment community located at {addr}." if b_name else f"Apartment complex at {addr}."
        desc_series.append(d)
        
    year_built = coalesce_cols(["year_built", "yearBuilt"])
    
    # Numeric sanitization & sensible defaults for apartment complexes
    baths_numeric = pd.to_numeric(full_baths, errors="coerce").fillna(0)
    beds_numeric = pd.to_numeric(beds, errors="coerce").fillna(2)
    baths_adjusted = np.where((baths_numeric == 0) & (beds_numeric >= 2), 2.0, np.where(baths_numeric == 0, 1.0, baths_numeric))
    
    lat_num = pd.to_numeric(latitude, errors="coerce")
    lon_num = pd.to_numeric(longitude, errors="coerce")
    
    out = pd.DataFrame({
        "formatted_address": formatted_address.astype(str).str.strip(),
        "latitude": lat_num,
        "longitude": lon_num,
        "list_price": pd.to_numeric(list_price, errors="coerce"),
        "beds": beds_numeric,
        "full_baths": baths_adjusted,
        "primary_photo": primary_photo,
        "alt_photos": alt_photo_col if alt_photo_col is not None else "",
        "property_url": property_url,
        "description": desc_series,
        "year_built": pd.to_numeric(year_built, errors="coerce")
    })
    
    # If coordinates are missing (as in maxcopell light mode), auto-geocode them on the fly
    missing_coords = out["latitude"].isna() | out["longitude"].isna()
    if missing_coords.any():
        print(f"🧭 Auto-geocoding {missing_coords.sum()} listing(s) missing coordinates in Apify export...")
        for idx in out[missing_coords].index:
            addr_q = out.loc[idx, "formatted_address"]
            coords = _geocode_safe_fallback(addr_q)
            if coords:
                out.loc[idx, "latitude"] = coords[0]
                out.loc[idx, "longitude"] = coords[1]
                
    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return out

def fetch_or_load_apify_zillow_dataset(
    source="cache/apify_zillow_export.json",
    apify_api_token=None,
    dataset_id=None
):
    """
    Loads Apify Zillow scraper data from either:
    1. A local JSON or CSV file path (e.g. `cache/apify_zillow_export.json` or `.csv`).
    2. Directly from Apify's Cloud Dataset API if `dataset_id` or `apify_api_token` is provided.
    
    Returns a cleaned, normalized DataFrame ready for spatial scoring and CLIP vision grading.
    """
    raw_df = None
    
    if dataset_id:
        url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=json"
        headers = {}
        if apify_api_token:
            headers["Authorization"] = f"Bearer {apify_api_token}"
        print(f"🌐 Fetching live Apify dataset from {url}...")
        try:
            resp = requests.get(url, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                raw_df = pd.DataFrame(resp.json())
                print(f"✅ Fetched {len(raw_df)} items directly from Apify Cloud API!")
            else:
                print(f"⚠️ Apify API returned status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️ Apify live fetch error: {e}")
            
    if raw_df is None and source and os.path.exists(source):
        print(f"📄 Loading Apify export from local file: {source}...")
        try:
            if source.endswith(".json"):
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_df = pd.DataFrame(data)
            elif source.endswith(".csv"):
                raw_df = pd.read_csv(source)
            elif source.endswith(".parquet"):
                raw_df = pd.read_parquet(source)
        except Exception as e:
            print(f"⚠️ Failed to read {source}: {e}")
            
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame()
        
    normalized_df = normalize_apify_zillow_dataframe(raw_df)
    print(f"🏢 Normalized {len(normalized_df)} valid Zillow listings with geographic coordinates from Apify!")
    return normalized_df

def run_apify_actor_and_get_dataset(actor_id, run_input, apify_api_token, timeout_secs=120):
    """
    Triggers an Apify actor run (e.g., 'parseforge/zillow-rentals-scraper', 'maxcopell/zillow-scraper', etc.)
    synchronously, waits for completion, and returns the resulting dataset items.
    """
    if not apify_api_token or not actor_id:
        return None
        
    formatted_actor_id = actor_id.replace("/", "~")
    run_url = f"https://api.apify.com/v2/acts/{formatted_actor_id}/runs?token={apify_api_token}"
    
    print(f"🚀 Triggering Apify Actor '{actor_id}'...")
    try:
        resp = requests.post(run_url, json=run_input, timeout=30.0)
        if resp.status_code not in (200, 201):
            print(f"⚠️ Failed to trigger Apify actor: {resp.status_code} {resp.text[:200]}")
            return None
            
        run_data = resp.json().get("data", {})
        run_id = run_data.get("id")
        dataset_id = run_data.get("defaultDatasetId")
        
        if not run_id:
            return None
            
        print(f"⏳ Waiting for Apify actor run {run_id} to complete (timeout: {timeout_secs}s)...")
        import time
        start_t = time.time()
        
        while time.time() - start_t < timeout_secs:
            time.sleep(5)
            status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={apify_api_token}"
            s_resp = requests.get(status_url, timeout=15.0)
            if s_resp.status_code == 200:
                s_data = s_resp.json().get("data", {})
                status = s_data.get("status")
                print(f"   • Apify run status: {status}")
                if status == "SUCCEEDED":
                    dataset_id = s_data.get("defaultDatasetId")
                    break
                elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    print(f"❌ Apify actor run ended with status: {status}")
                    return None
                    
        if dataset_id:
            return fetch_or_load_apify_zillow_dataset(apify_api_token=apify_api_token, dataset_id=dataset_id)
    except Exception as e:
        print(f"⚠️ Apify actor execution error: {e}")
    return None
