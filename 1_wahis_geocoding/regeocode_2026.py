import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

input_file = 'epi_2008_ONSSA_geocoding.xlsx'
output_file = 'epi_2008_geocoded_Nominatim.xlsx'

print(f"Reading {input_file}...")
df = pd.read_excel(input_file)

# Initialize geocoder
geolocator = Nominatim(user_agent="my_epi_project_2026")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

print("Starting geocoding process. This may take a few minutes...")

def get_location(row):
    # Construct address parts
    # Try specific to general
    parts = [
        str(row['Location']) if pd.notna(row['Location']) else '',
        str(row['COMMUNE']) if pd.notna(row['COMMUNE']) else '',
        str(row['PROVINCE']) if pd.notna(row['PROVINCE']) else '',
        str(row['REGION']) if pd.notna(row['REGION']) else '',
        'Morocco'
    ]
    # Filter empty parts
    parts = [p.strip() for p in parts if p.strip() and p.lower() != 'nan']
    address = ", ".join(parts)
    
    try:
        location = geocode(address)
        if location:
            return location.latitude, location.longitude, "Exact"
        
        # Fallback 1: Try without Location (specific village)
        if len(parts) > 1:
            address_fallback_1 = ", ".join(parts[1:])
            location = geocode(address_fallback_1)
            if location:
                return location.latitude, location.longitude, "Commune/Province Level"
        
        # Fallback 2: Try just Commune + Province + Morocco
        commune = str(row['COMMUNE']) if pd.notna(row['COMMUNE']) else ''
        province = str(row['PROVINCE']) if pd.notna(row['PROVINCE']) else ''
        if commune and province:
             address_fallback_2 = f"{commune}, {province}, Morocco"
             location = geocode(address_fallback_2)
             if location:
                 return location.latitude, location.longitude, "Commune Only"

        return None, None, "Not Found"
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        return None, None, "Error"

# Create new columns
df['New_Latitude'] = None
df['New_Longitude'] = None
df['Geocode_Status'] = None

total_rows = len(df)
for idx, row in df.iterrows():
    if idx % 10 == 0:
        print(f"Processing row {idx}/{total_rows}...")
    
    lat, lon, status = get_location(row)
    df.at[idx, 'New_Latitude'] = lat
    df.at[idx, 'New_Longitude'] = lon
    df.at[idx, 'Geocode_Status'] = status

print("Geocoding finished.")
print(df['Geocode_Status'].value_counts())

# Fill/Overwrite original columns if found, or keep old?
# User wants "new version", so maybe we replace them or offer new columns. 
# I'll save new columns primarily, but maybe I should update the main Lat/Lon if user wants just that.
# Let's update the main columns but keep backup in case.

df['Old_Latitude'] = df['Latitude']
df['Old_Longitude'] = df['Longitude']

# Update where found
mask = df['Geocode_Status'].isin(['Exact', 'Commune/Province Level', 'Commune Only'])
df.loc[mask, 'Latitude'] = df.loc[mask, 'New_Latitude']
df.loc[mask, 'Longitude'] = df.loc[mask, 'New_Longitude']

print(f"Saving to {output_file}...")
df.to_excel(output_file, index=False)
print("Done.")
