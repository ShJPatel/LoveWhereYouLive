import os
import json
import re
import requests
import numpy as np
import pandas as pd

def normalize_apify_zillow_dataframe(df):
    """
    Normalizes any raw pandas DataFrame extracted from Apify Zillow Scraper actors
    into the standard LoveWhereYouLive pipeline schema.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Handle nested address dict if present (e.g. from GraphQL/Zillow JSON)
    if "address" in df.columns:
        first_valid = df["address"].dropna().iloc[0] if len(df["address"].dropna()) > 0 else None
        if isinstance(first_valid, dict):
            addr_df = pd.json_normalize(df["address"])
            for c in addr_df.columns:
                df[f"addr_{c}"] = addr_df[c]
                
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
        
    street = coalesce_cols(["streetAddress", "street_address", "addr_streetAddress", "street", "streetName"])
    city = coalesce_cols(["city", "addr_city"], default="San Antonio")
    state = coalesce_cols(["state", "addr_state"], default="TX")
    zipcode = coalesce_cols(["zipcode", "zipCode", "addr_zipcode", "postalCode"], default="")
    
    raw_addr = coalesce_cols(["formatted_address", "address", "addr_address", "unparsedAddress"])
    synth_addr = street.astype(str) + ", " + city.astype(str) + ", " + state.astype(str)
    if not zipcode.isna().all() and (zipcode != "").any():
        synth_addr = synth_addr + " " + zipcode.astype(str)
    formatted_address = raw_addr.fillna(synth_addr)
    
    list_price = coalesce_cols(["list_price", "price", "unformattedPrice", "rentZestimate", "rate", "monthlyRent"])
    beds = coalesce_cols(["beds", "bedrooms", "bedroom", "minBeds"])
    full_baths = coalesce_cols(["full_baths", "bathrooms", "bathroomsFull", "baths", "bathroom"])
    latitude = coalesce_cols(["latitude", "lat", "addr_latitude"])
    longitude = coalesce_cols(["longitude", "lon", "lng", "addr_longitude"])
    
    primary_photo = coalesce_cols(["primary_photo", "imgSrc", "desktopWebHdpImageLink", "hiResImageLink", "image", "photo"])
    if "photos" in df.columns:
        def extract_first_photo(p):
            if isinstance(p, list) and len(p) > 0:
                first = p[0]
                if isinstance(first, dict):
                    return first.get("url") or first.get("mixedSources", {}).get("jpeg", [{}])[-1].get("url")
                return str(first)
            return None
        photos_first = df["photos"].apply(extract_first_photo)
        primary_photo = primary_photo.fillna(photos_first)
        
    property_url = coalesce_cols(["property_url", "url", "detailUrl", "hdpUrl", "webUrl"])
    description = coalesce_cols(["description", "homeDescription", "overview", "remarks"])
    year_built = coalesce_cols(["year_built", "yearBuilt"])
    
    out = pd.DataFrame({
        "formatted_address": formatted_address.astype(str).str.strip(),
        "latitude": pd.to_numeric(latitude, errors="coerce"),
        "longitude": pd.to_numeric(longitude, errors="coerce"),
        "list_price": pd.to_numeric(list_price, errors="coerce"),
        "beds": pd.to_numeric(beds, errors="coerce"),
        "full_baths": pd.to_numeric(full_baths, errors="coerce"),
        "primary_photo": primary_photo,
        "property_url": property_url,
        "description": description,
        "year_built": pd.to_numeric(year_built, errors="coerce")
    })
    
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
    
    # 1. Fetch live from Apify API if dataset_id provided
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
            
    # 2. Otherwise load from local file path if it exists
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
