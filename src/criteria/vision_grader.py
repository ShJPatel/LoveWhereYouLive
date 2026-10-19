import io
import os
import requests
from PIL import Image
import numpy as np
import pandas as pd

def score_photos_with_local_clip(
    df_top,
    max_candidates=25,
    liked_image_urls=None,
    disliked_image_urls=None,
    export_audit_html="outputs/visual_kitchen_audit.html"
):
    """
    Robust 3-Stage Local Vision Grader (OpenAI CLIP):
    1. Strict Room Isolation: Separates actual Kitchen photos from Living Room/Flooring photos
       (so a sunlit living room can NEVER mask an outdated kitchen via `max()`).
    2. 5-Axis Pairwise Feature Decomposition (Binary Softmax per physical attribute):
       - Axis A (Appliances): Stainless steel vs. White/black plastic or exposed electric coils
       - Axis B (Countertops): Thick quartz/granite stone slab vs. Thin laminate/formica/tile
       - Axis C (Cabinetry): Modern flat-panel/shaker cabinets vs. Dated 1990s oak/honey-wood cabinets
       - Axis D (Layout): Open-concept island/bar layout vs. Cramped closed galley wall kitchen
       - Axis E (Flooring): Hardwood / Luxury Vinyl Plank (LVP) vs. Wall-to-wall carpet
    3. Optional Few-Shot Personal Calibration:
       - If you pass 1-3 URLs of kitchens you LOVE (`liked_image_urls`) and 1-3 kitchens you HATE
         (`disliked_image_urls`), it computes exact cosine similarity in CLIP embedding space
         to your personal taste!
    4. Transparency Audit HTML:
       - Outputs `outputs/visual_kitchen_audit.html` showing the exact photo selected as the
         kitchen and every sub-score so you can immediately see why a unit scored what it did.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "transformers", "torch", "pillow"])
        from transformers import CLIPProcessor, CLIPModel
        import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Robust Multi-Axis CLIP Vision Engine on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Stage 1: Strict Room Classification
    room_labels = [
        "an indoor kitchen with stove, refrigerator, sink, countertops, and cabinets", # 0: Kitchen
        "an empty indoor living room or bedroom showing floors and walls",             # 1: Living/Bedroom
        "anindoor bathroom with toilet, shower, or bathtub",                           # 2: Bathroom
        "an outdoor building exterior, swimming pool, patio, street, or grass",        # 3: Exterior
        "a 2D floor plan diagram, map, or text logo"                                   # 4: Diagram
    ]

    # Stage 2: Independent Binary Contrasts for Kitchens (evaluated ONLY on verified kitchen photos)
    kitchen_axes = {
        "appliances": [
            "modern stainless steel kitchen appliances, sleek induction or gas range, stainless refrigerator",
            "old white or black plastic kitchen appliances, exposed electric coil burners, cheap stove"
        ],
        "counters": [
            "luxury thick quartz, marble, or polished granite stone kitchen countertops with undermount sink",
            "cheap laminate formica countertops,dated beige or speckled counter, or old drop-in sink"
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

    # Independent Binary Contrast for Flooring (evaluated on kitchen + living room photos)
    flooring_axis = [
        "clean modern luxury vinyl plank flooring, wide-plank hardwood floors, or modern large format tile",
        "wall-to-wall beige carpet, plush bedroom carpeting, or dated linoleum squares"
    ]

    def get_image_embedding(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]

    def binary_axis_prob(img, pair_prompts):
        inputs = processor(text=pair_prompts, images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
            probs = logits.softmax(dim=1).cpu().numpy()[0]
        return float(probs[0]) # Probability of the positive (modern) pole vs negative pole

    # Pre-compute user reference embeddings if provided (Few-Shot Personal Taste Anchors)
    liked_embs, disliked_embs = [], []
    if liked_image_urls:
        for u in liked_image_urls:
            try:
                im = Image.open(io.BytesIO(requests.get(u, timeout=4).content)).convert("RGB")
                liked_embs.append(get_image_embedding(im))
            except Exception:
                pass
    if disliked_image_urls:
        for u in disliked_image_urls:
            try:
                im = Image.open(io.BytesIO(requests.get(u, timeout=4).content)).convert("RGB")
                disliked_embs.append(get_image_embedding(im))
            except Exception:
                pass

    scores = []
    notes = []
    audit_rows = []

    for idx, row in df_top.head(max_candidates).iterrows():
        urls = []
        if pd.notna(row.get('primary_photo')) and str(row.get('primary_photo')).startswith('http'):
            urls.append(str(row['primary_photo']))
        if pd.notna(row.get('alt_photos')):
            alt = [u.strip() for u in str(row['alt_photos']).split(',') if u.strip().startswith('http')]
            urls.extend(alt[:14]) # Scan up to 15 photos so we don't miss the kitchen!

        # Deduplicate while preserving order
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))][:15]

        if not urls:
            scores.append(45.0)
            notes.append("No photo URLs (Unverified)")
            continue

        kitchen_candidates = [] # (kitchen_prob, url, img)
        living_candidates = []  # (living_prob, url, img)

        for url in urls:
            try:
                resp = requests.get(url, timeout=3.5)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                inputs = processor(text=room_labels, images=img, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    r_probs = model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]

                k_prob, l_prob, b_prob, ext_prob, diag_prob = [float(x) for x in r_probs]
                # Reject exteriors, bathrooms, and floorplan diagrams
                if ext_prob > 0.35 or diag_prob > 0.35 or b_prob > 0.40:
                    continue
                if k_prob >= 0.38:
                    kitchen_candidates.append((k_prob, url, img))
                elif l_prob >= 0.38:
                    living_candidates.append((l_prob, url, img))
            except Exception:
                continue

        # Sort by highest confidence that the photo is genuinely a kitchen
        kitchen_candidates.sort(key=lambda x: x[0], reverse=True)
        living_candidates.sort(key=lambda x: x[0], reverse=True)

        if not kitchen_candidates:
            # DO NOT reward a listing if it hides or lacks kitchen photos!
            floor_scores = [binary_axis_prob(im, flooring_axis) * 100.0 for _, _, im in living_candidates[:2]]
            fallback_floor = np.mean(floor_scores) if floor_scores else 45.0
            penalized = round(float(np.clip(fallback_floor * 0.65, 25.0, 58.0)), 1)
            scores.append(penalized)
            notes.append(f"No Kitchen Photo Found (Capped: {penalized}/100)")
            continue

        # Evaluate the Top 2 clearest kitchen photos across all 4 kitchen axes
        top_k = kitchen_candidates[:2]
        axis_results = {k: [] for k in kitchen_axes}
        floor_vals = []
        taste_vals = []

        for k_conf, k_url, k_img in top_k:
            for axis_name, pair_prompts in kitchen_axes.items():
                axis_results[axis_name].append(binary_axis_prob(k_img, pair_prompts) * 100.0)
            floor_vals.append(binary_axis_prob(k_img, flooring_axis) * 100.0)

            if liked_embs or disliked_embs:
                emb = get_image_embedding(k_img)
                pos_sim = max([float(np.dot(emb, le)) for le in liked_embs]) if liked_embs else 0.25
                neg_sim = max([float(np.dot(emb, de)) for de in disliked_embs]) if disliked_embs else 0.25
                # Convert cosine margin (-0.15 to +0.15) into 0-100 score
                margin = np.clip((pos_sim - neg_sim) * 350.0 + 50.0, 0.0, 100.0)
                taste_vals.append(float(margin))

        for _, _, l_img in living_candidates[:2]:
            floor_vals.append(binary_axis_prob(l_img, flooring_axis) * 100.0)

        # Aggregate per-axis scores (using mean across kitchen views, never `max()` across non-kitchens)
        app_s = float(np.mean(axis_results["appliances"]))
        cnt_s = float(np.mean(axis_results["counters"]))
        cab_s = float(np.mean(axis_results["cabinets"]))
        lay_s = float(np.mean(axis_results["layout"]))
        flr_s = float(np.mean(floor_vals)) if floor_vals else 50.0

        # Bottleneck / "Weakest Link" Rule:
        # If cabinets, counters, OR appliances are severely dated (< 35/100), cap or heavily penalize!
        raw_weighted = (0.28 * app_s) + (0.26 * cnt_s) + (0.24 * cab_s) + (0.12 * lay_s) + (0.10 * flr_s)
        weakest_kitchen_link = min(app_s, cnt_s, cab_s)
        if weakest_kitchen_link < 35.0:
            # Drag the score down toward the weakest link so old cabinets + new fridge don't get an 85
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

    # Export visual audit HTML so the user can inspect the exact kitchen photo & sub-scores
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
