import os
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PREF_PATH = "data/preferences.json"
HISTORY_PATH = "data/seen_listings_history.json"

def load_preferences(path=PREF_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        default_prefs = {
            "version": "4.3",
            "taste_profile": {
                "liked_kitchen_photo_urls": [],
                "disliked_kitchen_photo_urls": [],
                "liked_property_addresses": [],
                "favorites_notes": {},
                "rejected_property_addresses": [],
                "blacklisted_complex_keywords": [],
                "rejection_reasons_log": [],
                "excluded_grocery_chains": ["walmart", "dollar general", "family dollar", "target"],
                "dealbreaker_keywords": ["shared laundry", "coin-operated", "no washer", "student housing", "room for rent"]
            },
            "learned_weights_override": {}
        }
        with open(path, "w") as f:
            json.dump(default_prefs, f, indent=2)
        return default_prefs
    with open(path, "r") as f:
        prefs = json.load(f)
    # Ensure backwards-compatible keys exist
    tp = prefs.setdefault("taste_profile", {})
    tp.setdefault("favorites_notes", {})
    tp.setdefault("blacklisted_complex_keywords", [])
    return prefs

def save_preferences(prefs, path=PREF_PATH):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(prefs, f, indent=2)

def record_feedback(
    address=None,
    liked=False,
    kitchen_photo_url=None,
    reason="",
    blacklist_complex_name=None,
    new_dealbreaker_keyword=None,
    path=PREF_PATH
):
    """
    Updates your Persistent Favorites Shortlist, Long-Term Property/Complex Blacklist,
    and CLIP Visual Taste Anchors in `preferences.json`.
    """
    prefs = load_preferences(path)
    tp = prefs["taste_profile"]
    today = datetime.date.today().isoformat()

    if address:
        if liked:
            if address not in tp["liked_property_addresses"]:
                tp["liked_property_addresses"].append(address)
            if address in tp["rejected_property_addresses"]:
                tp["rejected_property_addresses"].remove(address)
            if reason:
                tp["favorites_notes"][address] = f"[{today}] {reason}"
        else:
            if address not in tp["rejected_property_addresses"]:
                tp["rejected_property_addresses"].append(address)
            if address in tp["liked_property_addresses"]:
                tp["liked_property_addresses"].remove(address)

    if blacklist_complex_name:
        cname = blacklist_complex_name.lower().strip()
        if cname and cname not in tp["blacklisted_complex_keywords"]:
            tp["blacklisted_complex_keywords"].append(cname)

    if kitchen_photo_url:
        if liked and kitchen_photo_url not in tp["liked_kitchen_photo_urls"]:
            tp["liked_kitchen_photo_urls"].append(kitchen_photo_url)
        elif not liked and kitchen_photo_url not in tp["disliked_kitchen_photo_urls"]:
            tp["disliked_kitchen_photo_urls"].append(kitchen_photo_url)

    if new_dealbreaker_keyword:
        kw = new_dealbreaker_keyword.lower().strip()
        if kw and kw not in tp["dealbreaker_keywords"]:
            tp["dealbreaker_keywords"].append(kw)

    if reason:
        tp["rejection_reasons_log"].append({
            "date": today,
            "address": address,
            "liked": liked,
            "kitchen_photo_url": kitchen_photo_url,
            "reason": reason
        })

    save_preferences(prefs, path)
    print(f"✅ Memory Updated ({path})!")
    print(f"  ⭐ Favorites Shortlist: {len(tp['liked_property_addresses'])} properties")
    print(f"  🚫 Blacklist: {len(tp['rejected_property_addresses'])} addresses | {len(tp['blacklisted_complex_keywords'])} complex names")
    print(f"  🖼️ Kitchen Vision Anchors: {len(tp['liked_kitchen_photo_urls'])} Liked | {len(tp['disliked_kitchen_photo_urls'])} Disliked")
    return prefs

def track_multi_week_changes_and_favorites(scored_df, prefs, history_path=HISTORY_PATH, fav_csv_path="outputs/favorites_shortlist.csv"):
    """
    Multi-Week Delta Tracker:
    - Tags every listing in `scored_df` with `listing_delta_status`:
      '⭐ FAVORITE', '🆕 NEW LISTING', '📉 PRICE DROP (-$X)', or 'Seen Previously'.
    - Exports your Favorites Shortlist to `outputs/favorites_shortlist.csv` so you have a clean tour sheet!
    """
    os.makedirs(os.path.dirname(history_path) if os.path.dirname(history_path) else ".", exist_ok=True)
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}

    today = datetime.date.today().isoformat()
    fav_set = set(prefs.get("taste_profile", {}).get("liked_property_addresses", []))
    fav_notes = prefs.get("taste_profile", {}).get("favorites_notes", {})

    statuses = []
    for _, row in scored_df.iterrows():
        addr = str(row.get("formatted_address", ""))
        price = float(row.get("list_price", 0))
        prefix = "⭐ FAVORITE | " if addr in fav_set else ""

        if addr not in history:
            statuses.append(prefix + "🆕 NEW")
            history[addr] = {"first_seen": today, "last_seen": today, "first_price": price, "last_price": price}
        else:
            prev_price = float(history[addr].get("last_price", price))
            history[addr]["last_seen"] = today
            if price < prev_price:
                drop = int(prev_price - price)
                statuses.append(prefix + f"📉 PRICE DROP (-${drop}/mo)")
                history[addr]["last_price"] = price
            elif price > prev_price:
                statuses.append(prefix + f"📈 Price Up (+${int(price - prev_price)}/mo)")
                history[addr]["last_price"] = price
            else:
                statuses.append(prefix + f"Seen ({history[addr].get('first_seen', today)})")

    scored_df = scored_df.copy()
    scored_df["listing_status"] = statuses
    scored_df["is_favorite"] = scored_df["formatted_address"].isin(fav_set)
    scored_df["favorite_notes"] = scored_df["formatted_address"].map(lambda a: fav_notes.get(a, ""))

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if fav_set:
        os.makedirs(os.path.dirname(fav_csv_path) if os.path.dirname(fav_csv_path) else ".", exist_ok=True)
        fav_df = scored_df[scored_df["is_favorite"]].copy()
        fav_df.to_csv(fav_csv_path, index=False)
        print(f"⭐ Exported {len(fav_df)} shortlisted favorites to {fav_csv_path}!")

    return scored_df

def calibrate_weights_from_ratings(scored_df, user_ratings_dict, base_weights, path=PREF_PATH):
    score_cols = list(base_weights.keys())
    X, y = [], []
    for addr_sub, target_score in user_ratings_dict.items():
        matches = scored_df[scored_df['formatted_address'].astype(str).str.contains(addr_sub, case=False, na=False)]
        if len(matches) > 0:
            row = matches.iloc[0]
            X.append([float(row.get(c, 50.0)) for c in score_cols])
            y.append(float(target_score))

    if len(X) < 3:
        print("Need at least 3 rated properties to auto-tune weights! Keeping current weights.")
        return base_weights

    model = Ridge(alpha=10.0, positive=True)
    model.fit(np.array(X), np.array(y))
    coefs = model.coef_
    if coefs.sum() > 0:
        learned = coefs / coefs.sum()
        new_weights = {}
        for col, lw in zip(score_cols, learned):
            new_weights[col] = round(float(0.50 * base_weights[col] + 0.50 * lw), 3)
        total = sum(new_weights.values())
        new_weights = {k: round(v / total, 3) for k, v in new_weights.items()}
        prefs = load_preferences(path)
        prefs["learned_weights_override"] = new_weights
        save_preferences(prefs, path)
        print("Learned new weights from your ratings:", new_weights)
        return new_weights
    return base_weights
