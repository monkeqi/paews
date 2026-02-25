"""
Quick diagnostic: check what coordinate ranges erdMWchla8day actually has.
"""
import requests
import json

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap"

# Check erdMWchla8day info
datasets = ["erdMWchla8day", "erdMH1chla8day", "erdMBchla8day"]

for dataset_id in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_id}")
    print(f"{'='*60}")
    
    url = f"{ERDDAP_BASE}/info/{dataset_id}/index.json"
    print(f"  Checking: {url[:70]}...")
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} — dataset may not exist")
            continue
        
        info = r.json()
        rows = info.get("table", {}).get("rows", [])
        
        # Find dimension info
        for row in rows:
            row_type = row[0] if len(row) > 0 else ""
            var_name = row[1] if len(row) > 1 else ""
            attr_name = row[2] if len(row) > 2 else ""
            data_type = row[3] if len(row) > 3 else ""
            value = row[4] if len(row) > 4 else ""
            
            # Print coordinate range info
            if attr_name in ("actual_range", "geospatial_lat_min", "geospatial_lat_max",
                             "geospatial_lon_min", "geospatial_lon_max",
                             "time_coverage_start", "time_coverage_end",
                             "title"):
                print(f"  {var_name}.{attr_name} = {value}")
            
            # Also print axis info
            if attr_name == "actual_range" and var_name in ("latitude", "longitude", "time", "altitude"):
                print(f"  >>> {var_name} range: {value}")
                
    except Exception as e:
        print(f"  Error: {e}")

# Also try a direct griddap metadata query
print(f"\n{'='*60}")
print("Trying direct DAS query for erdMWchla8day...")
print(f"{'='*60}")
das_url = f"{ERDDAP_BASE}/griddap/erdMWchla8day.das"
try:
    r = requests.get(das_url, timeout=30)
    if r.status_code == 200:
        # Look for latitude info in the DAS
        text = r.text
        # Find latitude section
        lat_start = text.find("latitude {")
        if lat_start >= 0:
            lat_section = text[lat_start:lat_start+500]
            print(f"  Latitude info:\n{lat_section[:300]}")
        
        lon_start = text.find("longitude {")
        if lon_start >= 0:
            lon_section = text[lon_start:lon_start+500]
            print(f"\n  Longitude info:\n{lon_section[:300]}")
            
        time_start = text.find("time {")
        if time_start >= 0:
            time_section = text[time_start:time_start+500]
            print(f"\n  Time info:\n{time_section[:300]}")
    else:
        print(f"  HTTP {r.status_code}")
except Exception as e:
    print(f"  Error: {e}")
