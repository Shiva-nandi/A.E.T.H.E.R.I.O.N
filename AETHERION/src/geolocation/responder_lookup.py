# ================================================================
# AETHERION
# Aerial Emergency Threat and Hazard Response
# Intelligence Operations Network
#
# "Guardian of the skies. Protector of lives."
#
# Author : P. Shiva Charan Reddy
# Type   : Personal Project
# Year   : 2026
# GitHub : github.com/ShivaCharanReddy/AETHERION
# ================================================================
"""
AETHERION — Responder Lookup & Geolocation
Finds nearest police station, hospital, coast guard, and lifeguard
using GPS coordinates.

Includes a Hyderabad-specific hardcoded database for demo/offline use,
with live Nominatim/Overpass API lookup as primary method.
"""

import math
import time
from typing import Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger('geolocation')


# ── HYDERABAD BEACH / COASTAL SAFETY DATABASE ─────────────────────────
# Primary demo database for BTech project (Hussain Sagar + coastal zones)
HYDERABAD_RESPONDERS = {
    "police_stations": [
        {"name": "Hussain Sagar PS",        "phone": "+914023232323", "lat": 17.4239, "lon": 78.4738, "zone": "lake"},
        {"name": "Necklace Road PS",         "phone": "+914023456789", "lat": 17.4205, "lon": 78.4600, "zone": "lake"},
        {"name": "Jubilee Hills PS",         "phone": "+914023355666", "lat": 17.4325, "lon": 78.4071, "zone": "urban"},
        {"name": "Banjara Hills PS",         "phone": "+914023370044", "lat": 17.4156, "lon": 78.4347, "zone": "urban"},
        {"name": "Chilkur PS",               "phone": "+914024528000", "lat": 17.3103, "lon": 78.3872, "zone": "reservoir"},
        {"name": "Sangareddy PS",            "phone": "+914025222222", "lat": 17.6279, "lon": 78.0882, "zone": "lake"},
    ],
    "hospitals": [
        {"name": "KIMS Hospital",            "phone": "+914044885000", "lat": 17.4239, "lon": 78.4481, "type": "trauma"},
        {"name": "Yashoda Hospitals",        "phone": "+914045677777", "lat": 17.4400, "lon": 78.3489, "type": "trauma"},
        {"name": "Apollo Hospitals Jubilee", "phone": "+914023607777", "lat": 17.4325, "lon": 78.4071, "type": "trauma"},
        {"name": "Osmania General Hospital", "phone": "+914024600177", "lat": 17.3808, "lon": 78.4740, "type": "govt"},
        {"name": "Nizam Institute (NIMS)",   "phone": "+914023310009", "lat": 17.3970, "lon": 78.4600, "type": "trauma"},
        {"name": "Care Hospitals",           "phone": "+914040683333", "lat": 17.4451, "lon": 78.3615, "type": "trauma"},
    ],
    "coast_guard": [
        {"name": "Telangana Lake Patrol",    "phone": "+914027777777", "lat": 17.4239, "lon": 78.4738, "zone": "all"},
        {"name": "Hussain Sagar Patrol",     "phone": "+914023000111", "lat": 17.4250, "lon": 78.4755, "zone": "lake"},
    ],
    "lifeguards": [
        {"name": "Necklace Road Lifeguards", "phone": "+919876543210", "lat": 17.4205, "lon": 78.4610, "zone": "lake"},
        {"name": "Lumbini Park Guards",      "phone": "+919876543211", "lat": 17.4243, "lon": 78.4717, "zone": "lake"},
        {"name": "Sanjeevaiah Park Guards",  "phone": "+919876543212", "lat": 17.4375, "lon": 78.4867, "zone": "lake"},
        {"name": "Chilkur Reservoir Guards", "phone": "+919876543213", "lat": 17.3103, "lon": 78.3872, "zone": "reservoir"},
    ],
}


class ResponderLookup:
    """
    Finds nearest responders to a given GPS location.
    Priority: Live Nominatim/Overpass API → Hyderabad hardcoded DB → Default numbers.
    """

    def __init__(self, hyderabad_mode: bool = True, use_live_api: bool = True):
        self.hyderabad_mode = hyderabad_mode
        self.use_live_api = use_live_api
        self._cache: Dict[str, dict] = {}

    def get_nearest(self, lat: float, lon: float,
                    responder_types: List[str]) -> Dict[str, dict]:
        """
        Returns dict of {responder_type: contact_info} for the nearest
        responder of each requested type.
        """
        result = {}

        type_map = {
            'police_station': 'police_stations',
            'hospital':       'hospitals',
            'coast_guard':    'coast_guard',
            'lifeguard':      'lifeguards',
        }

        for r_type in responder_types:
            if r_type == 'public_alert':
                result['public_alert'] = {'name': 'Public Broadcast', 'phone': ''}
                continue

            db_key = type_map.get(r_type)
            if not db_key:
                continue

            # Try live API first
            if self.use_live_api:
                contact = self._live_lookup(lat, lon, r_type)
                if contact:
                    result[r_type] = contact
                    continue

            # Fallback to Hyderabad DB
            if self.hyderabad_mode and db_key in HYDERABAD_RESPONDERS:
                contact = self._nearest_from_db(
                    lat, lon, HYDERABAD_RESPONDERS[db_key]
                )
                if contact:
                    result[r_type] = contact

        return result

    def _nearest_from_db(self, lat: float, lon: float,
                          db: List[dict]) -> Optional[dict]:
        if not db:
            return None
        best = min(db, key=lambda r: self._haversine(lat, lon, r['lat'], r['lon']))
        dist_km = self._haversine(lat, lon, best['lat'], best['lon']) / 1000
        return {**best, 'distance_km': round(dist_km, 2)}

    def _live_lookup(self, lat: float, lon: float, r_type: str) -> Optional[dict]:
        """Query OpenStreetMap Overpass API for nearest amenity."""
        cache_key = f"{r_type}_{lat:.3f}_{lon:.3f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        amenity_map = {
            'police_station': 'police',
            'hospital':       'hospital',
            'coast_guard':    'coast_guard',
            'lifeguard':      'lifeguard',
        }
        amenity = amenity_map.get(r_type, 'police')

        try:
            import urllib.request, json
            radius = 10000  # 10km search radius
            query = f"""
            [out:json][timeout:10];
            node[amenity={amenity}](around:{radius},{lat},{lon});
            out 1;
            """
            url = "https://overpass-api.de/api/interpreter"
            req = urllib.request.Request(
                url,
                data=query.encode(),
                method='POST',
                headers={'Content-Type': 'text/plain'}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())

            if data.get('elements'):
                el = data['elements'][0]
                tags = el.get('tags', {})
                contact = {
                    'name':    tags.get('name', f'Nearest {r_type}'),
                    'phone':   tags.get('phone', tags.get('contact:phone', '')),
                    'lat':     el.get('lat', lat),
                    'lon':     el.get('lon', lon),
                    'source':  'overpass_api',
                    'distance_km': round(self._haversine(lat, lon, el.get('lat', lat), el.get('lon', lon)) / 1000, 2),
                }
                self._cache[cache_key] = contact
                return contact
        except Exception as e:
            logger.debug(f"Live lookup failed ({r_type}): {e}")

        return None

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Distance between two GPS points in meters."""
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def get_all_responders_for_zone(self, zone: str = 'lake') -> dict:
        """Return all responders for a given zone (for pre-mission briefing)."""
        result = {}
        for db_key, contacts in HYDERABAD_RESPONDERS.items():
            result[db_key] = [c for c in contacts if c.get('zone') in (zone, 'all')]
        return result
