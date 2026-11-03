"""Static NYC reference data used by the Streamlit UI.

The model only knows the 217 neighbourhoods it was trained on, so the UI
offers them as dropdowns (grouped by borough) instead of free text.
"""

BOROUGH_HOODS = {
    "Manhattan": [
        "Battery Park City", "Chelsea", "Chinatown", "Civic Center", "East Harlem",
        "East Village", "Financial District", "Flatiron District", "Gramercy",
        "Greenwich Village", "Harlem", "Hell's Kitchen", "Inwood", "Kips Bay",
        "Little Italy", "Lower East Side", "Marble Hill", "Midtown",
        "Morningside Heights", "Murray Hill", "NoHo", "Nolita", "Roosevelt Island",
        "SoHo", "Stuyvesant Town", "Theater District", "Tribeca", "Two Bridges",
        "Upper East Side", "Upper West Side", "Washington Heights", "West Village",
    ],
    "Brooklyn": [
        "Bath Beach", "Bay Ridge", "Bedford-Stuyvesant", "Bensonhurst", "Bergen Beach",
        "Boerum Hill", "Borough Park", "Brighton Beach", "Brooklyn Heights",
        "Brownsville", "Bushwick", "Canarsie", "Carroll Gardens", "Clinton Hill",
        "Cobble Hill", "Columbia St", "Coney Island", "Crown Heights", "Cypress Hills",
        "DUMBO", "Downtown Brooklyn", "Dyker Heights", "East Flatbush",
        "East New York", "Flatbush", "Flatlands", "Fort Greene", "Fort Hamilton",
        "Gowanus", "Gravesend", "Greenpoint", "Kensington", "Manhattan Beach",
        "Midwood", "Mill Basin", "Navy Yard", "Park Slope", "Prospect Heights",
        "Prospect-Lefferts Gardens", "Red Hook", "Sea Gate", "Sheepshead Bay",
        "South Slope", "Sunset Park", "Vinegar Hill", "Williamsburg", "Windsor Terrace",
    ],
    "Queens": [
        "Arverne", "Astoria", "Bay Terrace", "Bayside", "Bayswater", "Belle Harbor",
        "Bellerose", "Breezy Point", "Briarwood", "Cambria Heights", "College Point",
        "Corona", "Ditmars Steinway", "Douglaston", "East Elmhurst", "Edgemere",
        "Elmhurst", "Far Rockaway", "Flushing", "Forest Hills", "Fresh Meadows",
        "Glendale", "Hollis", "Holliswood", "Howard Beach", "Jackson Heights",
        "Jamaica", "Jamaica Estates", "Jamaica Hills", "Kew Gardens",
        "Kew Gardens Hills", "Laurelton", "Little Neck", "Long Island City",
        "Maspeth", "Middle Village", "Neponsit", "Ozone Park", "Queens Village",
        "Rego Park", "Richmond Hill", "Ridgewood", "Rockaway Beach", "Rosedale",
        "South Ozone Park", "Springfield Gardens", "St. Albans", "Sunnyside",
        "Whitestone", "Woodhaven", "Woodside",
    ],
    "Bronx": [
        "Allerton", "Baychester", "Belmont", "Bronxdale", "Castle Hill", "City Island",
        "Claremont Village", "Clason Point", "Co-op City", "Concourse",
        "Concourse Village", "East Morrisania", "Eastchester", "Edenwald", "Fieldston",
        "Fordham", "Highbridge", "Hunts Point", "Kingsbridge", "Longwood", "Melrose",
        "Morris Heights", "Morris Park", "Morrisania", "Mott Haven", "Mount Eden",
        "Mount Hope", "North Riverdale", "Norwood", "Olinville", "Parkchester",
        "Pelham Bay", "Pelham Gardens", "Port Morris", "Riverdale", "Schuylerville",
        "Soundview", "Spuyten Duyvil", "Throgs Neck", "Tremont", "Unionport",
        "University Heights", "Van Nest", "Wakefield", "West Farms",
        "Westchester Square", "Williamsbridge", "Woodlawn",
    ],
    "Staten Island": [
        "Arden Heights", "Arrochar", "Bay Terrace, Staten Island", "Bull's Head",
        "Castleton Corners", "Clifton", "Concord", "Dongan Hills", "Eltingville",
        "Emerson Hill", "Graniteville", "Grant City", "Great Kills", "Grymes Hill",
        "Howland Hook", "Huguenot", "Mariners Harbor", "Midland Beach", "New Brighton",
        "New Dorp", "New Dorp Beach", "New Springville", "Oakwood", "Port Richmond",
        "Prince's Bay", "Randall Manor", "Rosebank", "Rossville", "Shore Acres",
        "Silver Lake", "South Beach", "St. George", "Stapleton", "Todt Hill",
        "Tompkinsville", "Tottenville", "West Brighton", "Westerleigh", "Willowbrook",
    ],
}

BOROUGH_CENTER = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}

# Approximate centres for well-known neighbourhoods. Anything not listed here
# falls back to its borough centre; users can still fine-tune lat/long.
HOOD_COORDS = {
    "Midtown": (40.7549, -73.9840), "Hell's Kitchen": (40.7638, -73.9918),
    "Upper West Side": (40.7870, -73.9754), "Upper East Side": (40.7736, -73.9566),
    "Harlem": (40.8116, -73.9465), "East Harlem": (40.7957, -73.9389),
    "East Village": (40.7265, -73.9815), "West Village": (40.7358, -74.0036),
    "Chelsea": (40.7465, -74.0014), "Lower East Side": (40.7150, -73.9843),
    "SoHo": (40.7233, -74.0030), "Tribeca": (40.7163, -74.0086),
    "Financial District": (40.7075, -74.0089), "Chinatown": (40.7158, -73.9970),
    "Greenwich Village": (40.7336, -74.0027), "Murray Hill": (40.7479, -73.9757),
    "Gramercy": (40.7368, -73.9845), "Kips Bay": (40.7423, -73.9801),
    "Theater District": (40.7590, -73.9845), "Roosevelt Island": (40.7620, -73.9500),
    "Morningside Heights": (40.8100, -73.9626), "Washington Heights": (40.8417, -73.9393),
    "Inwood": (40.8677, -73.9212),
    "Williamsburg": (40.7081, -73.9571), "Bushwick": (40.6944, -73.9213),
    "Bedford-Stuyvesant": (40.6872, -73.9418), "Greenpoint": (40.7304, -73.9515),
    "Park Slope": (40.6710, -73.9814), "Crown Heights": (40.6694, -73.9422),
    "DUMBO": (40.7033, -73.9881), "Brooklyn Heights": (40.6960, -73.9936),
    "Fort Greene": (40.6890, -73.9742), "Clinton Hill": (40.6897, -73.9660),
    "Sunset Park": (40.6453, -74.0124), "Flatbush": (40.6410, -73.9594),
    "Coney Island": (40.5755, -73.9707), "Red Hook": (40.6734, -74.0083),
    "Gowanus": (40.6733, -73.9880), "Bay Ridge": (40.6263, -74.0299),
    "Prospect Heights": (40.6775, -73.9692), "Cobble Hill": (40.6860, -73.9962),
    "Carroll Gardens": (40.6795, -73.9991), "Windsor Terrace": (40.6561, -73.9800),
    "Astoria": (40.7644, -73.9235), "Long Island City": (40.7447, -73.9485),
    "Flushing": (40.7675, -73.8331), "Jamaica": (40.7027, -73.7890),
    "Ridgewood": (40.7043, -73.9018), "Forest Hills": (40.7185, -73.8448),
    "Elmhurst": (40.7370, -73.8801), "Jackson Heights": (40.7557, -73.8831),
    "Sunnyside": (40.7433, -73.9196),
    "Riverdale": (40.8900, -73.9126), "Fordham": (40.8613, -73.8900),
    "Mott Haven": (40.8087, -73.9229),
    "St. George": (40.6437, -74.0735),
}


def coords_for(borough: str, hood: str) -> tuple[float, float]:
    return HOOD_COORDS.get(hood, BOROUGH_CENTER[borough])
