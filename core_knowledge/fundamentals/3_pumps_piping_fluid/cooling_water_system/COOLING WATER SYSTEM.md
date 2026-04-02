# MARINE COOLING WATER SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Central Cooling System (Seawater & Freshwater Circuits - HT/LT)

**Folder Name:** cooling_water_system

**Prepared by:** Senior Marine Engineer & Thermodynamic Systems Expert (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Heat Rejection

## 1.1 Thermodynamic Basis of Central Cooling
Marine engines convert fuel energy into mechanical power, but approximately **30–35% of that energy** must be rejected as waste heat.
*   **The Physics:** $Q = \dot{m} C_p \Delta T$
    Where $Q$ is the heat removed, $\dot{m}$ is the mass flow rate, and $C_p$ is the specific heat capacity of the coolant. 
*   **Central Cooling Concept:** To prevent internal corrosion from seawater, ships use a "Closed Loop" Freshwater system to cool the engine, which is then cooled by Seawater in a "Central Heat Exchanger."

## 1.2 HT vs. LT Circuit Physics
1.  **High Temperature (HT) Circuit:** Operates at 80°C – 90°C. It cools the most sensitive components (Cylinder Liners, Heads). High temperature is required to prevent "Cold Corrosion" from sulfuric acid condensation.
2.  **Low Temperature (LT) Circuit:** Operates at 35°C – 45°C. It cools the Lube Oil, Charge Air, and auxiliary systems.

## 1.3 Pump Cavitation Physics
Cooling pumps move massive volumes of water.
*   **The Physics:** If the pressure at the pump suction falls below the water's vapor pressure, "bubbles" form and collapse violently. 
*   **Result:** Pitting of the impeller and a catastrophic drop in flow. This is why "Sea Chest" location and venting are critical.

---

# Part 2 – Major Components & System Layout

## 2.1 Sea Chests and Strainers
*   **Sea Chests:** Recesses in the hull where seawater enters. Typically two: "High Sea Chest" (for shallow/dirty water) and "Low Sea Chest" (for deep sea/clean water).
*   **Sea Strainers:** Coarse filters to catch plastic, seaweed, and fish.

## 2.2 Central Heat Exchangers (PHE)
*   **Type:** Usually Titanium Plate Heat Exchangers (PHE). 
*   **Physics:** Corrugated plates create "Turbulent Flow" at low velocities, significantly increasing the heat transfer coefficient ($U$).

## 2.3 Circulating Pumps
*   **Seawater (SW) Pumps:** Large centrifugal pumps.
*   **Freshwater (FW) Pumps:** Dedicated pumps for the HT and LT loops.

## 2.4 Thermostatic Valves (3-Way Valves)
*   **Function:** Automatically mix "Hot Return" water with "Cold Supply" water to maintain a rock-steady temperature at the engine inlet.

## 2.5 Expansion Tank
*   **Function:** Acts as a header tank to provide "Static Head" (pressure) and to allow for the thermal expansion of the water.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Pressure Differential:** A healthy PHE has a clear pressure drop (e.g., 0.5 bar SW side). If the drop increases, the cooler is fouled.
*   **Chemical Balance:** Freshwater should be treated with corrosion inhibitors (Nitrite/Borate) to maintain a protective film on the engine liners.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **HT Outlet Temp (Engine)** | 80°C – 85°C | > 95°C (Shutdown) |
| **LT Inlet Temp** | 36°C – 40°C | > 45°C (Poor LO Cooling) |
| **SW Pump Discharge Press** | 2.5 – 3.5 bar | < 1.5 bar (Pump Trip) |
| **FW Nitrite Level** | 1500 – 3000 ppm | < 1000 ppm (Corrosion) |
| **Expansion Tank Level** | 50% – 70% | < 30% (Low Level Alarm) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Bio-Fouling (Barnacles and Slime)
*   **Symptom:** SW pressure rises, LT temperature climbs even with pumps at full speed.
*   **Root Cause:** Marine growth inside the sea chest or PHE plates.
*   **Solution:** Check the **MGPS (Marine Growth Protection System)** anodes (Copper/Aluminum).

## 4.2 Galvanic Corrosion (Pitting)
*   **Symptom:** Small "pinhole" leaks in the PHE plates.
*   **Root Cause:** Failure of the "Sacrificial Anodes" or mixing incompatible metals (e.g., Copper pipes and Aluminum components).

## 4.3 Air-Lock
*   **Symptom:** One cylinder head is significantly hotter than others; "High Temp" alarm on a single unit.
*   **Root Cause:** Air trapped in the high-point jacket after maintenance.
*   **Physics:** Air is a thermal insulator; it prevents water from reaching the hot metal.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Temperature Spread" Trick
*   **Trade Trick:** To find a fouled cooler without a borescope, measure the **Temperature Difference ($\Delta T$)** of the SW across the cooler. 
    *   **High $\Delta T$ + Low Flow:** Fouled on the SW side.
    *   **Low $\Delta T$ + High Flow:** Fouled on the FW side (oil/scale buildup).

## 5.2 Cleaning Sea Strainers "Under Pressure"
*   **Expert Insight:** Always ensure the "Air Vent" on the strainer is open when re-flooding it. If you open the main valve with air inside, you will create a **Water Hammer** that can blow out the PHE gaskets.

## 5.3 Emergency Cooling (The Cross-Over)
*   **Trade Trick:** Most ships have a cross-over valve between the Fire Main and the SW Cooling system. In an emergency (total pump failure), you can use the Fire Pump to provide temporary SW cooling.

## 5.4 Detecting a Liner Leak via the Expansion Tank
*   **Expert Insight:** If the Expansion Tank is "bubbling" or smells of exhaust, you have a cracked cylinder liner or a failed head gasket. The combustion pressure is leaking into the cooling water.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Routine Checks
*   **Chemical Analysis:** Test the FW for Nitrite and Chloride levels weekly.
*   **Anodes:** Inspect and clean MGPS anodes every 3 months.

## 6.2 Cleaning the PHE (Plate Heat Exchanger)
*   **Method:** "Back-flushing" with air/water can remove loose debris. For hard scale, the PHE must be opened and chemically cleaned with a weak acid solution (Citric Acid).

## 6.3 Pump Seal Replacement
*   **Method:** When replacing a mechanical seal, ensure the "Sleeve" is polished. A single scratch will destroy the new carbon face within hours.

---

# Part 7 – Miscellaneous Knowledge

*   **Freshwater Generator (FWG):** Uses the "Waste Heat" from the HT circuit to evaporate seawater in a vacuum, producing fresh drinking water for the crew.
*   **Pre-Heater:** When the ship is in port, the HT water is kept warm (60°C) by a steam or electric pre-heater to ensure a safe "Warm Start" for the engine.

**End of Document**
