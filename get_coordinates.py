from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import random
import time

def get_coordinates(location_name="Downtown", num_points=10):
    geolocator = Nominatim(user_agent="my_agent")
    
    try:
        location = geolocator.geocode(location_name + " Boston")
        if location is None:
            return []
        
        base_lat, base_lon = location.latitude, location.longitude
        
        coordinates = []
        for _ in range(num_points):
            lat = base_lat + random.uniform(-0.01, 0.01)
            lon = base_lon + random.uniform(-0.01, 0.01)
            coordinates.append([lat, lon])
        
        time.sleep(1)
        
        return coordinates
    
    except (GeocoderTimedOut, GeocoderServiceError):
        print(f"Error: Unable to geocode {location_name}")
        return []
