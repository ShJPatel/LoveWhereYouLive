import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PREF_PATH = "data/preferences.json"

def load_preferences(path=PREF_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        default_prefs = {
            "version": "4.0",
            "taste_profile": {
                "liked_kitchen_photo_urls": [],
                "disliked_kitchen_photo_urls": [],
                "liked_property_addresses": [],
                "rejected_property_addresses": [],
                "rejection_reasons_log": [],
                "excluded_grocery_chains": ["walmart", "dollar general", "family dollar"],
                "dealbreaker_keywords": ["shared laundry", "coin-operated", "no washer", "student housing", "room for rent"]
            },
            "learned_weights_override": {}
        }
        with open(path, "w") as f:
            json.dump(default_prefs, f, indent=2)
        return default_prefs
    with open(path, "r") as f:
        return json.load(f)

def save_preferences(prefs, path=PREF_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(prefs, f, indent=2)

def record_feedback(
    address=None,
    liked=False,
    kitchen_photo_url=None,
    reason="",
    new_dealbreaker_keyword=None,
    path=PREF_PATH
):
    """Records thumbs-up / thumbs-down feedback on a property or kitchen photo into persistent JSON memory."""
    prefs = load_preferences(path)
    tp = prefs["taste_profile"]

    if address:
        if liked and address not in tp["liked_property_addresses"]:
            tp["liked_property_addresses"].append(address)
        elif not liked and address not in tp["rejected_property_addresses"]:
            tp["rejected_property_addresses"].append(address)

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
            "address": address,
            "liked": liked,
            "kitchen_photo_url": kitchen_photo_url,
            "reason": reason
        })

    save_preferences(prefs, path)
    print(f"Memory updated! Total Liked Properties: {len(tp['liked_property_addresses'])}, Rejected: {len(tp['rejected_property_addresses'])}")
    print(f"Kitchen Anchors -> Liked: {len(tp['liked_kitchen_photo_urls'])}, Disliked: {len(tp['disliked_kitchen_photo_urls'])}")
    return prefs

def calibrate_weights_from_ratings(scored_df, user_ratings_dict, base_weights, path=PREF_PATH):
    """
    Active Learning Weight Tuner:
    Given a dictionary of {address_substring: user_score_0_to_100}, fits a Ridge regression
    over the sub-scores to learn which criteria actually drive your personal satisfaction,
    and saves the updated weights into `data/preferences.json`.
    """
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
        # Blend 50% prior config weights + 50% learned empirical weights for stability
        new_weights = {}
        for col, lw in zip(score_cols, learned):
            new_weights[col] = round(float(0.50 * base_weights[col] + 0.50 * lw), 3)
        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        new_weights = {k: round(v / total, 3) for k, v in new_weights.items()}
        prefs = load_preferences(path)
        prefs["learned_weights_override"] = new_weights
        save_preferences(prefs, path)
        print("Learned new weights from your ratings:", new_weights)
        return new_weights
    return base_weights
