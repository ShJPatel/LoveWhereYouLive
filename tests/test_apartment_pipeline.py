import os
import time
import json
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Import backend modules assuming repo root is in path
try:
    from src.criteria import ACTIVE_CRITERIA
    from src.preference_memory import load_preferences, record_feedback
    from src.inspector import address_from_url_or_query
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.criteria import ACTIVE_CRITERIA
    from src.preference_memory import load_preferences, record_feedback
    from src.inspector import address_from_url_or_query

class TestApartmentHunterPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.benchmarks = {}
        cls.cache_dir = "cache"
        os.makedirs(cls.cache_dir, exist_ok=True)
        cls.log_path = os.path.join(cls.cache_dir, "unit_test_benchmarks.json")

    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        elapsed = time.time() - self.start_time
        test_name = self._testMethodName
        self.benchmarks[test_name] = round(elapsed, 4)
        print(f"⏱️ [{test_name}] executed in {elapsed:.4f} seconds.")

    @classmethod
    def tearDownClass(cls):
        with open(cls.log_path, "w") as f:
            json.dump(cls.benchmarks, f, indent=2)
        print(f"📊 Task expense benchmarks saved to {cls.log_path}")

    def test_01_url_address_parser(self):
        """Test URL slug parsing for Realtor.com, Zillow, and raw queries."""
        realtor_url = "https://www.realtor.com/rentals/details/806-W-French-Pl_San-Antonio_TX_78212_M70795-64780"
        parsed = address_from_url_or_query(realtor_url)
        self.assertEqual(parsed, "806 W French Pl San Antonio TX 78212")

        raw_query = "6220 Culebra Rd, San Antonio, TX"
        self.assertEqual(address_from_url_or_query(raw_query), raw_query)

    def test_02_persistent_memory_loading(self):
        """Test persistent JSON preference loading and safety defaults."""
        prefs = load_preferences()
        self.assertIn("taste_profile", prefs)
        self.assertIn("dealbreaker_keywords", prefs["taste_profile"])
        self.assertIn("shared laundry", prefs["taste_profile"]["dealbreaker_keywords"])

    def test_03_criteria_functions_mock_row(self):
        """Test individual scoring criteria against a mock rental listing row."""
        mock_row = pd.Series({
            "description": "Gorgeous 2-bed apartment with open-concept kitchen, stainless steel appliances, granite countertops, luxury vinyl plank flooring, in-unit washer and dryer, and water softener loop.",
            "year_built": 2022,
            "latitude": 29.4485,
            "longitude": -98.6105,
            "utm_x": 535000.0,
            "utm_y": 3257000.0,
            "parking_garage": 1.0
        })
        
        mock_ctx = {
            "config": {
                "workplace": {"coords": (29.4485, -98.6105)},
                "costco_locations": [{"name": "Costco NW", "coords": (29.57, -98.60)}],
                "cutoffs": {"strict_appliance_text_match": False}
            },
            "trees": {
                "grocery": (MagicMock(), ["H-E-B"]),
                "park": (MagicMock(), ["Park"]),
                "highway": (MagicMock(),),
                "railway": (MagicMock(),)
            }
        }
        mock_ctx["trees"]["grocery"][0].query.return_value = (500.0, 0)
        mock_ctx["trees"]["park"][0].query.return_value = (400.0, 0)
        mock_ctx["trees"]["highway"][0].query.return_value = (300.0, 0)
        mock_ctx["trees"]["railway"][0].query.return_value = (1000.0, 0)

        # Run criteria
        results = {}
        for fn in ACTIVE_CRITERIA:
            results.update(fn(mock_row, mock_ctx))

        self.assertTrue(results.get("passes_appliance_gate"))
        self.assertGreaterEqual(results.get("modern_interior_score", 0), 70.0)
        self.assertGreaterEqual(results.get("water_softener_score", 0), 75.0)
        self.assertEqual(results.get("property_age_years"), 4) # 2026 - 2022 = 4

    def test_04_feedback_recording_and_complex_blacklist(self):
        """Test recording favorites and complex-wide blacklists in memory."""
        test_pref_path = "cache/test_preferences.json"
        if os.path.exists(test_pref_path):
            os.remove(test_pref_path)

        record_feedback(
            address="999 Test Ave, San Antonio, TX",
            liked=False,
            blacklist_complex_name="Villas at Test",
            reason="Dated interior",
            path=test_pref_path
        )
        
        prefs = load_preferences(test_pref_path)
        self.assertIn("villas at test", prefs["taste_profile"]["blacklisted_complex_keywords"])
        self.assertIn("999 Test Ave, San Antonio, TX", prefs["taste_profile"]["rejected_property_addresses"])
        
        if os.path.exists(test_pref_path):
            os.remove(test_pref_path)

if __name__ == "__main__":
    unittest.main()
