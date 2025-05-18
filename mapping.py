import json
import os

def get_analytics_data():
    """
    Loads metrics and plot paths for all models for analytics dashboard.
    Returns a dictionary suitable for passing to the analytics.html template.
    """
    base_metrics = "analytics"
    maps_folder = "maps"
    models = [
        ("crop_recommendation", "Crop Recommendation"),
        ("crop_price", "Crop Price"),
        ("fertilizer", "Fertilizer"),
        ("disease", "Disease"),
        ("breast_cancer", "Breast Cancer"),
    ]
    analytics_data = {}
    for key, display_name in models:
        metrics_path = os.path.join(base_metrics, f"{key}_metrics.json")
        # Collect all PNGs for this model
        maps_list = []
        if os.path.isdir(maps_folder):
            for fname in os.listdir(maps_folder):
                if fname.startswith(key + "_") and fname.endswith(".png"):
                    maps_list.append(f"maps/{fname}")
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
        analytics_data[key] = {
            "display_name": display_name,
            "metrics": metrics,
            "maps": maps_list
        }
    return analytics_data
