import json
from datetime import date
import secrets
from pathlib import Path

REGISTRY = Path("data/notebooks.json")

def generate_id():
    return "nb_" + secrets.token_hex(6)

CATEGORIES = {
  "1_propulsion_main_machinery": [
      "main_engine", "diesel_alternator", "emergency_generator", "propeller_shafting", 
      "reduction_gear", "bow_thruster", "stern_thruster", "steering_gear", 
      "engine_control_system", "alarm_monitoring_system"
  ],
  "2_fuel_oil_purification": [
      "fuel_oil_purifier", "lube_oil_purifier", "fuel_oil_system", "lube_oil_system", 
      "bunker_system", "oily_water_separator"
  ],
  "3_pumps_piping_fluid": [
      "fire_pump", "emergency_fire_pump", "bilge_pump", "ballast_pump", 
      "cargo_pump", "cooling_water_system", "fresh_water_generator", 
      "hydrophore_system", "sewage_treatment_plant"
  ],
  "4_electrical_power_distribution": [
      "main_switchboard", "emergency_switchboard", "motor_control_center", 
      "transformers", "lighting_system"
  ],
  "5_boiler_steam": [
      "auxiliary_boiler", "exhaust_gas_economizer", "steam_turbines", 
      "feed_water_system", "condensate_system"
  ],
  "6_compressed_air_gas": [
      "main_air_compressor", "starting_air_compressor", "control_air_compressor", 
      "working_air_compressor", "inert_gas_generator", "nitrogen_generator"
  ],
  "7_cooling_fresh_water_refrigeration": [
      "central_cooling_system", "sea_water_cooling_system", "fresh_water_cooling_system", 
      "ac_plant_refrigeration", "provision_refrigeration"
  ],
  "8_waste_pollution_control": [
      "incinerator", "garbage_compactor", "ballast_water_treatment", "scrubber_system"
  ],
  "9_deck_machinery_cargo_handling": [
      "windlass", "mooring_winch", "cargo_cranes", "hatch_covers", "cargo_pumps_deck"
  ],
  "10_bridge_navigation_equipment": [
      "radar", "ecdis", "gyro_compass", "autopilot", "speed_log", "echo_sounder", 
      "gps", "ais", "vdr", "gmdss"
  ],
  "11_firefighting_safety_lifesaving": [
      "co2_system", "water_mist_system", "foam_system", "fire_detection_system", 
      "lifeboats", "rescue_boats", "liferafts", "scba", "eeBD"
  ]
}

def fill_notebooks():
    try:
        with REGISTRY.open("r", encoding="utf-8") as f:
            notebooks = json.load(f)
    except Exception:
        notebooks = []

    existing = {n.get("name") for n in notebooks}

    for cat_id, sub_items in CATEGORIES.items():
        for item in sub_items:
            # Create a pretty name
            name = item.replace("_", " ").title()
            # The User's list had specific capitalizations like 'DA', but Title() will be fine
            if item == "diesel_alternator":
                name = "Diesel Alternator (DA)"
            
            if name not in existing:
                nb = {
                    "id": generate_id(),
                    "name": name,
                    "equipment_category": cat_id,
                    "created": str(date.today())
                }
                notebooks.append(nb)
                existing.add(name)

    # Clean duplicates or old unnamed models if needed
    
    with REGISTRY.open("w", encoding="utf-8") as f:
        json.dump(notebooks, f, indent=2)

    print(f"Created/Validated {sum(len(v) for v in CATEGORIES.values())} notebooks.")

if __name__ == "__main__":
    fill_notebooks()
