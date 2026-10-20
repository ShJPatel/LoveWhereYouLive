import io
import os
import pickle
import requests
from PIL import Image
import numpy as np
import pandas as pd

def score_photos_with_local_clip(
    df_top,
    max_candidates=25,
    liked_image_urls=None,
    disliked_image_urls=None,
    export_audit_html="outputs/visual_kitchen_audit.html",
    cache_path="cache/clip_photo_cache.pkl",
    force_refresh_vision=False
):
    """
    Robust 3-Stage Local Vision Grader (OpenAI CLIP) with Persistent Per-URL Disk Caching:
    - Caches room classification, 5-axis scores, and 512-d image embeddings per photo URL in `cache_path`.
    - If a photo URL was already downloaded & evaluated once, it never re-downloads or re-runs CLIP on it!
    - Even better: when you add new `liked_image_urls` or `disliked_image_urls` in Step 6, it computes
      new taste similarity scores in <0.05 seconds directly from the cached 512-d embeddings!
    """
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    photo_cache = {}
    if os.path.exists(cache_path) and not force_refresh_vision:
        try:
            with open(cache_path, "rb") as f:
                photo_cache = pickle.load(f)
            print(f"⚡ Loaded CLIP vision cache with {len(photo_cache)} pre-analyzed photo URLs from {cache_path}")
        except Exception:
            photo_cache = {}

    # Lazy-load CLIP model ONLY if we encounter an uncached URL!
    model_state = {"model": None, "processor": None, "device": None}

    def ensure_clip_loaded():
        if model_state["model"] is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
            except ImportError:
                import subprocess
                subprocess.check_call(["pip", "install", "-q", "transformers", "torch", "pillow"])
                from transformers import CLIPProcessor, CLIPModel
                import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⏳ Loading Local CLIP Vision Engine on {device} for uncached photos...")
            model_state["device"] = device
            model_state["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            model_state["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    room_labels = [
        "an indoor kitchen with stove, refrigerator, sink, countertops, and cabinets", # 0: Kitchen
        "an empty indoor living room or bedroom showing floors and walls",             # 1: Living/Bedroom
        "an indoor bathroom with toilet, shower, or bathtub",                          # 2: Bathroom
        "an outdoor building exterior, swimming pool, patio, street, or grass",        # 3: Exterior
        "a 2D floor plan diagram, map, or text logo"                                   # 4: Diagram
    ]

    kitchen_axes = {
        "appliances": [
            "modern stainless steel kitchen appliances, sleek induction or gas range, stainless refrigerator",
            "old white or black plastic kitchen appliances, exposed electric coil burners, cheap stove"
        ],
        "counters": [
            "luxury thick quartz, marble, or polished granite stone kitchen countertops with undermount sink",
            "cheap laminate formica countertops, dated beige or speckled counter, or old drop-in sink"
        ],
        "cabinets": [
            "modern shaker style or sleek flat-panel contemporary kitchen cabinets with modern hardware",
            "dated 1980s or 1990s orange oak wood cabinets, cheap builder-grade brown wood doors, or worn cabinets"
        ],
        "layout": [
            "spacious open-concept kitchen with a large center island or breakfast bar overlooking the living room",
            "cramped narrow galley kitchen, closed-off small apartment kitchen with no island and tight walls"
        ]
    }

    flooring_axis = [
        "clean modern luxury vinyl plank flooring, wide-plank hardwood floors, or modern large format tile",
        "wall-to-wall beige carpet, plush bedroom carpeting, or dated linoleum squares"
    ]

    cache_dirty = [False]

    def analyze_or_get_cached_url(url):
        """Returns cached dict for a photo URL, or downloads & runs CLIP once and caches it."""
        if url in photo_cache and not force_refresh_vision:
            return photo_cache[url]

        ensure_clip_loaded()
        import torch
        model = model_state["model"]
        processor = model_state["processor"]
        device = model_state["device"]

        try:
            resp = requests.get(url, timeout=3.5)
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")

            # 1. Image embedding (normalized 512-d vector)
            img_inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**img_inputs)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            emb = feat.cpu().numpy()[0]

            # 2. Room classification
            r_inputs = processor(text=room_labels, images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                r_probs = model(**r_inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
            k_prob, l_prob, b_prob, ext_prob, diag_prob = [float(x) for x in r_probs]

            entry = {
                "valid": True,
                "emb": emb,
                "k_prob": k_prob,
                "l_prob": l_prob,
                "b_prob": b_prob,
                "ext_prob": ext_prob,
                "diag_prob": diag_prob,
                "axes": {},
                "floor": 50.0
            }

            is_rejected_room = (ext_prob > 0.35 or diag_prob > 0.35 or b_prob > 0.40)
            if not is_rejected_room and (k_prob >= 0.38 or l_prob >= 0.38):
                # Compute flooring score
                f_inputs = processor(text=flooring_axis, images=img, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    f_prob = model(**f_inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0][0]
                entry["floor"] = float(f_prob * 100.0)

                if k_prob >= 0.38:
                    for axis_name, pair_prompts in kitchen_axes.items():
                        ax_inputs = processor(text=pair_prompts, images=img, return_tensors="pt", padding=True).to(device)
                        with torch.no_grad():
                            ax_prob = model(**ax_inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0][0]
                        entry["axes"][axis_name] = float(ax_prob * 100.0)

            photo_cache[url] = entry
            cache_dirty[0] = True
            return entry
        except Exception:
            entry = {"valid": False}
            photo_cache[url] = entry
            cache_dirty[0] = True
            return entry

    # Pre-fetch embeddings for user's Liked / Disliked reference kitchen URLs (also cached!)
    liked_embs, disliked_embs = [], []
    if liked_image_urls:
        for u in liked_image_urls:
            res = analyze_or_get_cached_url(u)
            if res.get("valid") and "emb" in res:
                liked_embs.append(res["emb"])
    if disliked_image_urls:
        for u in disliked_image_urls:
            res = analyze_or_get_cached_url(u)
            if res.get("valid") and "emb" in res:
                disliked_embs.append(res["emb"])

    scores = []
    notes = []
    audit_rows = []
    cache_hits = 0
    cache_misses = 0

    for idx, row in df_top.head(max_candidates).iterrows():
        urls = []
        if pd.notna(row.get('primary_photo')) and str(row.get('primary_photo')).startswith('http'):
            urls.append(str(row['primary_photo']))
        if pd.notna(row.get('alt_photos')):
            alt = [u.strip() for u in str(row['alt_photos']).split(',') if u.strip().startswith('http')]
            urls.extend(alt[:14])

        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))][:15]

        if not urls:
            scores.append(45.0)
            notes.append("No photo URLs (Unverified)")
            continue

        kitchen_candidates = [] # (k_prob, url, entry)
        living_candidates = []  # (l_prob, url, entry)

        for url in urls:
            if url in photo_cache and not force_refresh_vision:
                cache_hits += 1
            else:
                cache_misses += 1
            entry = analyze_or_get_cached_url(url)
            if not entry.get("valid"):
                continue

            if entry["ext_prob"] > 0.35 or entry["diag_prob"] > 0.35 or entry["b_prob"] > 0.40:
                continue
            if entry["k_prob"] >= 0.38 and entry.get("axes"):
                kitchen_candidates.append((entry["k_prob"], url, entry))
            elif entry["l_prob"] >= 0.38:
                living_candidates.append((entry["l_prob"], url, entry))

        kitchen_candidates.sort(key=lambda x: x[0], reverse=True)
        living_candidates.sort(key=lambda x: x[0], reverse=True)

        if not kitchen_candidates:
            floor_scores = [e["floor"] for _, _, e in living_candidates[:2]]
            fallback_floor = np.mean(floor_scores) if floor_scores else 45.0
            penalized = round(float(np.clip(fallback_floor * 0.65, 25.0, 58.0)), 1)
            scores.append(penalized)
            notes.append(f"No Kitchen Photo Found (Capped: {penalized}/100)")
            continue

        top_k = kitchen_candidates[:2]
        axis_results = {k: [] for k in kitchen_axes}
        floor_vals = []
        taste_vals = []

        for k_conf, k_url, k_entry in top_k:
            for axis_name in kitchen_axes:
                axis_results[axis_name].append(k_entry["axes"][axis_name])
            floor_vals.append(k_entry["floor"])

            if liked_embs or disliked_embs:
                emb = k_entry["emb"]
                pos_sim = max([float(np.dot(emb, le)) for le in liked_embs]) if liked_embs else 0.25
                neg_sim = max([float(np.dot(emb, de)) for de in disliked_embs]) if disliked_embs else 0.25
                margin = np.clip((pos_sim - neg_sim) * 350.0 + 50.0, 0.0, 100.0)
                taste_vals.append(float(margin))

        for _, _, l_entry in living_candidates[:2]:
            floor_vals.append(l_entry["floor"])

        app_s = float(np.mean(axis_results["appliances"]))
        cnt_s = float(np.mean(axis_results["counters"]))
        cab_s = float(np.mean(axis_results["cabinets"]))
        lay_s = float(np.mean(axis_results["layout"]))
        flr_s = float(np.mean(floor_vals)) if floor_vals else 50.0

        raw_weighted = (0.28 * app_s) + (0.26 * cnt_s) + (0.24 * cab_s) + (0.12 * lay_s) + (0.10 * flr_s)
        weakest_kitchen_link = min(app_s, cnt_s, cab_s)
        if weakest_kitchen_link < 35.0:
            raw_weighted = 0.50 * raw_weighted + 0.50 * weakest_kitchen_link

        if taste_vals:
            raw_weighted = 0.70 * raw_weighted + 0.30 * float(np.mean(taste_vals))

        final_v_score = round(float(np.clip(raw_weighted, 10.0, 100.0)), 1)
        scores.append(final_v_score)

        breakdown_str = (
            f"Kitchen {final_v_score}/100 "
            f"[App:{round(app_s)} | Cnt:{round(cnt_s)} | Cab:{round(cab_s)} | Open:{round(lay_s)} | Flr:{round(flr_s)}]"
        )
        notes.append(breakdown_str)

        audit_rows.append({
            "address": row.get("formatted_address", "Listing"),
            "price": row.get("list_price", ""),
            "url": row.get("property_url", "#"),
            "kitchen_img_url": top_k[0][1],
            "kitchen_conf": round(top_k[0][0] * 100),
            "final_score": final_v_score,
            "appliances": round(app_s),
            "counters": round(cnt_s),
            "cabinets": round(cab_s),
            "layout": round(lay_s),
            "flooring": round(flr_s)
        })

    if cache_dirty[0]:
        with open(cache_path, "wb") as f:
            pickle.dump(photo_cache, f)
        print(f"✅ Saved {len(photo_cache)} analyzed photo embeddings to {cache_path} (Hits: {cache_hits}, New: {cache_misses})")
    else:
        print(f"⚡ 100% CLIP Cache Hit ({cache_hits} photos)! Evaluated all kitchens & personal taste vectors in <0.1s!")

    if export_audit_html and audit_rows:
        os.makedirs(os.path.dirname(export_audit_html), exist_ok=True)
        cards_html = ""
        for r in sorted(audit_rows, key=lambda x: x["final_score"], reverse=True):
            badge_color = "#1b7e3c" if r["final_score"] >= 75 else ("#b57c00" if r["final_score"] >= 55 else "#b82e2e")
            cards_html += f"""
            <div style="border:1px solid #ddd; border-radius:8px; width:340px; overflow:hidden; background:#fff; box-shadow:0 2px 5px rgba(0,0,0,0.08);">
                <img src="{r['kitchen_img_url']}" style="width:100%; height:210px; object-fit:cover;" />
                <div style="padding:12px; font-family:sans-serif; font-size:13px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="background:{badge_color}; color:#fff; padding:3px 8px; border-radius:4px; font-weight:bold;">
                            Kitchen Score: {r['final_score']}/100
                        </span>
                        <b>${r['price']}/mo</b>
                    </div>
                    <div style="margin:6px 0; font-weight:bold;">{r['address']}</div>
                    <div style="color:#444; line-height:1.5;">
                        • <b>Appliances (Stainless vs Coil):</b> {r['appliances']}/100<br>
                        • <b>Counters (Stone vs Laminate):</b> {r['counters']}/100<br>
                        • <b>Cabinets (Modern vs Dated Oak):</b> {r['cabinets']}/100<br>
                        • <b>Open Island Layout:</b> {r['layout']}/100<br>
                        • <b>Hard Flooring (No Carpet):</b> {r['flooring']}/100
                    </div>
                    <div style="margin-top:8px;">
                        <a href="{r['url']}" target="_blank" style="color:#0066cc; text-decoration:none; font-weight:bold;">Open Listing →</a>
                    </div>
                </div>
            </div>
            """
        html_doc = f"""<!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>Kitchen Visual Audit Report</title></head>
        <body style="background:#f5f6f8; padding:24px; font-family:sans-serif;">
            <h2>🍳 Multi-Axis Kitchen & Interior Visual Audit Report</h2>
            <p>Showing the exact photo classified as the <b>Kitchen</b> for each candidate and its 5-axis sub-scores (weakest-link penalized):</p>
            <div style="display:flex; flex-wrap:wrap; gap:16px;">{cards_html}</div>
        </body></html>"""
        with open(export_audit_html, "w") as f:
            f.write(html_doc)
        print(f"Saved Visual Kitchen Audit card grid to {export_audit_html}!")

    df_top = df_top.copy()
    df_top.loc[df_top.head(max_candidates).index, 'vision_modern_score'] = scores
    df_top.loc[df_top.head(max_candidates).index, 'vision_notes'] = notes
    return df_top
