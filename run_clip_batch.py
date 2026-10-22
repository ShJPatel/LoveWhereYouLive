import yaml
import pandas as pd
from src.pipeline import run_pipeline

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("🚀 Launching pipeline with local CLIP vision grader...")
    print(f"Config filters: {config.get('filters')}")
    print(f"Config cutoffs: {config.get('cutoffs')}")

    scored_df = run_pipeline(
        config_dict=config,
        cache_dir="cache",
        pref_file="data/preferences.json",
        run_local_clip=True,
        clip_max_candidates=150,
        force_refresh_scrape=False,
        enforce_cutoffs=True
    )

    print(f"🏁 Complete! Ranked {len(scored_df)} candidates.")
    
    # Sync into review companion sheet as well
    out_review = "outputs/san_antonio_ranked_rentals_review.csv"
    if len(scored_df) > 0:
        scored_df.to_csv(out_review, index=False)
        print(f"✅ Updated {out_review}")
