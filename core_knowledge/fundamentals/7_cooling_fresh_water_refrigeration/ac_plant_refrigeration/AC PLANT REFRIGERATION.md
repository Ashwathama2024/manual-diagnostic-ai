# MARINE AC PLANT REFRIGERATION – COMPLETE CORE KNOWLEDGE

**Equipment:** Accommodation Air Conditioning (AC) Plant (e.g., Daikin, Carrier, HI-PRES, Novenco)

**Folder Name:** ac_plant_refrigeration

**Prepared by:** Senior Marine Engineer & HVAC Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Refrigeration

## 1.1 The Physics of the Vapor Compression Cycle
Refrigeration is the process of moving heat from a cold space (the Accommodation) to a warm space (the Ocean).
*   **The Physics:** It relies on the relationship between **Pressure and Boiling Point**. By changing the pressure of a refrigerant gas (e.g., R404a, R134a), we can force it to boil at very low temperatures (absorbing heat) and condense at high temperatures (releasing heat).
*   **The Four Stages:**
    1.  **Compression:** Increases pressure and temperature ($P \uparrow, T \uparrow$).
    2.  **Condensation:** High-pressure gas turns to liquid, releasing heat to seawater ($Q_{out}$).
    3.  **Expansion:** Rapid pressure drop through a nozzle ($P \downarrow$).
    4.  **Evaporation:** Low-pressure liquid turns to gas, absorbing heat from the room air ($Q_{in}$).

## 1.2 Superheat and Subcooling Physics
*   **Superheat:** The temperature rise of the gas *after* it has completely evaporated. **Physics:** Ensures no liquid droplets enter the compressor (which would cause "Liquid Slugging"). Target is usually 5°C – 8°C.
*   **Subcooling:** The temperature drop of the liquid *after* it has completely condensed. **Physics:** Ensures only 100% liquid reaches the expansion valve, improving efficiency.

## 1.3 Latent vs. Sensible Heat
*   **Sensible Heat:** Changes the temperature of the air (making it "Cool").
*   **Latent Heat:** Removes moisture (moisture in the air condenses on the cold coils). This is the "Dehumidification" role of the AC, which is vital for crew comfort in the tropics.

---

# Part 2 – Major Components & System Layout

## 2.1 The Compressor
*   **Reciprocating:** Uses pistons. Good for varying loads.
*   **Screw:** Uses rotors. More efficient and quieter for large AC plants.
*   **Capacity Control:** Uses "Slide Valves" or "Unloaders" to match the cooling output to the ship's internal heat load.

## 2.2 The Condenser
A shell-and-tube heat exchanger. Seawater flows through the tubes, and high-pressure refrigerant gas fills the shell.

## 2.3 Thermostatic Expansion Valve (TEV / TXV)
The "Brain" of the cycle.
*   **Physics:** It modulates the refrigerant flow based on the "Superheat" at the evaporator outlet. It uses a sensing bulb to detect temperature and a diaphragm to move the valve needle.

## 2.4 Air Handling Unit (AHU)
The large box where the ship's air is filtered, cooled, and sometimes heated. It contains the **Evaporator Coils** and the **Supply Fans**.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Sight Glass:** Full of clear liquid. No bubbles (bubbles indicate low gas or a leak).
*   **Sound:** Steady hum from the compressor. No "Rattling" (Liquid slugging) or "Screeching" (Bearing fail).
*   **Pressures:** Stable suction and discharge pressures based on the refrigerant type and SW temperature.

## 3.2 Typical Operating Parameters (R404a Example)

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Suction Pressure** | 3.5 – 5.0 bar | < 2.0 bar (Low Gas/Ice) |
| **Discharge Pressure** | 12 – 18 bar | > 22 bar (High SW Temp) |
| **Superheat** | 5°C – 8°C | < 2°C (Slugging Risk) |
| **Supply Air Temp** | 12°C – 15°C | > 20°C (Poor Cooling) |
| **Oil Level** | 1/2 Sight Glass | < 1/4 (Compressor Risk) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Icing Up" of the Evaporator
*   **Symptom:** Air flow from the vents drops; ice visible on the AHU coils.
*   **Root Cause:** Dirty air filters or a failed blower belt.
*   **Physics:** Without enough air flow, the refrigerant doesn't absorb enough heat. The coil temperature drops below 0°C, and moisture in the air freezes on the fins.

## 4.2 High Discharge Pressure (HP Trip)
*   **Symptom:** Compressor stops; "HP Trip" alarm.
*   **Root Cause:** Fouled condenser tubes (SW side) or a "Non-condensable" (Air) in the system.
*   **Physics:** If the heat cannot be rejected to the seawater, the refrigerant pressure climbs until the safety switch trips.

## 4.3 Low Suction Pressure (LP Trip)
*   **Symptom:** Compressor stops; "LP Trip" alarm.
*   **Root Cause:** Refrigerant leak or a blocked expansion valve (moisture frozen in the nozzle).

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Temperature Difference" Test
*   **Trade Trick:** Touch the pipes before and after the **Liquid Line Filter Drier**. They should be the same temperature. **If the outlet is cooler than the inlet, the filter is blocked.** The pressure drop is causing the refrigerant to expand and cool inside the filter.

## 5.2 Detecting "Air" in the System
*   **Trade Trick:** If the discharge pressure is high but the seawater is cold and the condenser is clean, you have **Air** in the system. 
*   **Method:** Perform a "Pump-down." Let the system sit for 1 hour. Compare the refrigerant pressure to its saturation temperature (using a P-T chart). **If the pressure is higher than the chart says it should be, you have non-condensable gasses.**

## 5.3 The "Ice-Block" Valve Test
*   **Trade Trick:** If you suspect the expansion valve is blocked by ice (moisture in gas), wrap a hot rag around the valve body for 2 minutes. If the suction pressure rises instantly, you have moisture. **Solution:** Change the Filter-Drier core.

## 5.4 Checking Oil "Acidity"
*   **Expert Insight:** If a compressor burns out, it creates **Acid** in the oil. **Trade Trick:** Always use an "Acid Test Kit" on the oil before installing a new compressor. If you don't, the acid in the pipes will eat the new motor's windings in weeks.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Filters:** Clean the AHU air filters. **A dirty filter is the #1 cause of AC complaints.**
*   **Leaks:** Scan all joints with a portable electronic leak detector.

## 6.2 Monthly Condenser Cleaning
*   **Method:** "Back-flush" the seawater side of the condenser. If the ship is in sandy or silty water, rod through the tubes to maintain heat transfer.

## 6.3 Quarterly Refrigerant Log
*   **Requirement:** Record all refrigerant additions in the "F-Gas Log" (or equivalent). This is a legal requirement to track global warming potential (GWP) gasses.

---

# Part 7 – Miscellaneous Knowledge

*   **Evacuation (Vacuuming):** If the system is opened, you must pull a vacuum down to **500 microns**. This is not just to remove air, but to "boil off" any moisture. Moisture reacts with refrigerant to form **Hydrofluoric Acid**.
*   **Safety:** Always wear goggles and gloves when handling refrigerant. A "Liquid Burn" can cause permanent eye damage or frostbite instantly.

**End of Document**
