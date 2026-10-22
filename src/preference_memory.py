import os
import json
import pandas as pd

def load_preferences(path="data/preferences.json"):
    """Loads persistent preference memory, blacklist definitions, and tracked custom complexes."""
    default_prefs = {
        "taste_profile": {
            "liked_property_addresses": [],
            "rejected_property_addresses": [],
            "blacklisted_complex_keywords": [],
            "liked_kitchen_photo_urls": [],
            "disliked_kitchen_photo_urls": [],
            "tracked_custom_listings": []
        },
        "feedback_log": []
    }
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "taste_profile" not in data:
                data["taste_profile"] = default_prefs["taste_profile"]
            else:
                for k, v in default_prefs["taste_profile"].items():
                    if k not in data["taste_profile"]:
                        data["taste_profile"][k] = v
            if "feedback_log" not in data:
                data["feedback_log"] = []
            return data
        except Exception:
            pass
    return default_prefs

def register_custom_listing(listing_dict, path="data/preferences.json"):
    """
    Persists an unlisted/custom complex (e.g. from Zillow /apartments/ or Step 7 Inspector)
    into `prefs['taste_profile']['tracked_custom_listings']` so every future `run_pipeline()`
    automatically injects, scores, and ranks it alongside MLS listings.
    """
    if not listing_dict or not listing_dict.get("formatted_address"):
        return None
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    prefs = load_preferences(path)
    tracked = prefs["taste_profile"].get("tracked_custom_listings", [])
    
    addr_clean = str(listing_dict["formatted_address"]).strip().lower()
    url_clean = str(listing_dict.get("property_url", "")).strip().lower()
    
    # Upsert existing entry if matched by address or URL
    updated = False
    for idx, item in enumerate(tracked):
        ex_addr = str(item.get("formatted_address", "")).strip().lower()
        ex_url = str(item.get("property_url", "")).strip().lower()
        if (addr_clean and addr_clean == ex_addr) or (url_clean and url_clean.startswith("http") and url_clean == ex_url):
            tracked[idx] = {**item, **{k: v for k, v in listing_dict.items() if v is not None}}
            updated = True
            break
            
    if not updated:
        tracked.append(listing_dict)
        
    prefs["taste_profile"]["tracked_custom_listings"] = tracked
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
    return prefs

def record_feedback(
    address=None,
    liked=True,
    reason=None,
    blacklist_complex_name=None,
    liked_kitchen_url=None,
    disliked_kitchen_url=None,
    custom_listing=None,
    path="data/preferences.json"
):
    """
    Records persistent taste and listing feedback:
    - Pins liked properties to `liked_property_addresses`
    - Adds complex names to `blacklisted_complex_keywords`
    - Stores reference kitchen photo URLs for CLIP cosine embedding comparisons
    - Optionally registers custom/unlisted apartment complexes via `custom_listing`
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    prefs = load_preferences(path)
    
    if address:
        addr_clean = str(address).strip()
        if liked:
            if addr_clean not in prefs["taste_profile"]["liked_property_addresses"]:
                prefs["taste_profile"]["liked_property_addresses"].append(addr_clean)
            if "rejected_property_addresses" in prefs["taste_profile"] and addr_clean in prefs["taste_profile"]["rejected_property_addresses"]:
                prefs["taste_profile"]["rejected_property_addresses"].remove(addr_clean)
        else:
            if addr_clean in prefs["taste_profile"]["liked_property_addresses"]:
                prefs["taste_profile"]["liked_property_addresses"].remove(addr_clean)
            if "rejected_property_addresses" not in prefs["taste_profile"]:
                prefs["taste_profile"]["rejected_property_addresses"] = []
            if addr_clean not in prefs["taste_profile"]["rejected_property_addresses"]:
                prefs["taste_profile"]["rejected_property_addresses"].append(addr_clean)
            
    if blacklist_complex_name:
        kw = str(blacklist_complex_name).strip().lower()
        if kw and kw not in prefs["taste_profile"]["blacklisted_complex_keywords"]:
            prefs["taste_profile"]["blacklisted_complex_keywords"].append(kw)
            
    if liked_kitchen_url:
        u = str(liked_kitchen_url).strip()
        if u and u not in prefs["taste_profile"]["liked_kitchen_photo_urls"]:
            prefs["taste_profile"]["liked_kitchen_photo_urls"].append(u)
            
    if disliked_kitchen_url:
        u = str(disliked_kitchen_url).strip()
        if u and u not in prefs["taste_profile"]["disliked_kitchen_photo_urls"]:
            prefs["taste_profile"]["disliked_kitchen_photo_urls"].append(u)
            
    prefs["feedback_log"].append({
        "address": address,
        "liked": liked,
        "reason": reason,
        "blacklist_complex_name": blacklist_complex_name,
        "liked_kitchen_url": liked_kitchen_url,
        "disliked_kitchen_url": disliked_kitchen_url
    })
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)
        
    if custom_listing:
        prefs = register_custom_listing(custom_listing, path=path)
        
    return prefs
