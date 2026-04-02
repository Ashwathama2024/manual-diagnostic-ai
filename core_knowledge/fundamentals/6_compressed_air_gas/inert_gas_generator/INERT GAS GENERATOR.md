# MARINE INERT GAS GENERATOR (IGG) – COMPLETE CORE KNOWLEDGE

**Equipment:** Combustion-type Inert Gas Generator (e.g., Alfa Laval Smit, Wärtsilä Moss, KangRim)

**Folder Name:** inert_gas_generator

**Prepared by:** Senior Marine Engineer & Tanker Safety Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Inerting

## 1.1 The Physics of the "Fire Triangle"
The primary purpose of the IGG is to eliminate one leg of the fire triangle: **Oxygen**. 
*   **The Requirement:** Hydrocarbon vapors cannot ignite if the Oxygen ($O_2$) content is below **8%** by volume. SOLAS requires IGG systems to deliver gas with less than **5% $O_2$** to provide a safety margin.
*   **The Physics:** The IGG burns marine fuel in a controlled furnace. The combustion process uses up the oxygen in the air, replacing it with Nitrogen ($N_2$) and Carbon Dioxide ($CO_2$).

## 1.2 Stoichiometric Combustion Physics
*   **The Physics:** To achieve < 5% $O_2$, the air-to-fuel ratio must be precisely controlled. 
*   **Excess Air:** If there is too much air, the $O_2$ levels rise above 8% (Dangerous).
*   **Incomplete Combustion:** If there is too little air, the burner produces Carbon Monoxide ($CO$) and Soot, which can contaminate the cargo and foul the scrubbers.

## 1.3 Gas Cooling and Dew Point Physics
Combustion gasses leave the furnace at $> 1000^\circ C$ and are saturated with water vapor.
*   **Scrubbing Physics:** The gas passes through a seawater "Scrubber Tower" which cools the gas to within 2°C of the seawater temperature and removes Sulfur Oxides ($SO_x$).
*   **Dew Point:** The gas must be "Dried" (using a refrigerated dryer) to a dew point of approx. $-20^\circ C$ before entering the tanks to prevent condensation and corrosion.

---

# Part 2 – Major Components & Safety Barriers

## 2.1 The Burner and Combustion Chamber
*   **Horizontal/Vertical Furnace:** Water-cooled jacketed chamber where fuel is burned at high pressure.
*   **Combustion Air Blowers:** High-capacity fans that provide the air for burning and for "Purging" the system.

## 2.2 The Scrubber Tower
*   **Function:** Cools the gas and removes soot/acid.
*   **Physics:** Uses "Packed Bed" or "Venturi" stages where seawater is sprayed. The alkalinity of the seawater neutralizes the sulfuric acid formed during combustion.

## 2.3 The Deck Water Seal (The #1 Safety Barrier)
*   **Physics:** A literal "U-tube" of water that allows Inert Gas to flow *to* the tanks but prevents explosive cargo vapors from flowing *back* to the Engine Room. It is the physical equivalent of a check valve that cannot "seize" or "spark."

## 2.4 Pressure/Vacuum (P/V) Breaker
*   **Function:** A large liquid-filled trap on deck.
*   **Physics:** Protects the cargo tanks from structural damage. If the IGG pressure is too high, it "blows" out. If the tank vacuum is too low (e.g., during discharge), it "sucks" air in to prevent a tank collapse.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Oxygen Content:** Stable at 2% – 4%.
*   **Gas Temperature:** Cool to the touch at the deck main.
*   **Deck Seal:** Constant water flow and level.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Oxygen ($O_2$) Content** | 2.0% – 4.5% | > 5% (Alarm) / > 8% (Stop) |
| **Discharge Pressure** | 0.1 – 0.2 bar | Low Press (Auto Stop) |
| **Gas Outlet Temp** | < 35°C | > 45°C (High Temp Trip) |
| **Scrubber SW Press** | 2.5 – 4.0 bar | Low Press (Burner Trip) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 High Oxygen Content
*   **Symptom:** $O_2$ analyzer shows 6%+, burner shuts down.
*   **Root Cause:** Faulty air/fuel ratio controller or a leak in the blower suction manifold (sucking in fresh air).
*   **Expert Insight:** Often caused by a worn fuel pump plunger that isn't delivering the commanded amount of oil.

## 4.2 Scrubber "Sooting" (Carry-over)
*   **Symptom:** Black water coming out of the scrubber drain; soot detected in cargo tanks.
*   **Root Cause:** Poor atomization at the burner or the scrubber "demister" is missing or damaged.

## 4.3 Deck Seal Water Loss
*   **Symptom:** Low level alarm on the deck water seal.
*   **Root Cause:** Heating coil leak (in cold climates) or the SW supply pump has failed. **DANGER:** This allows cargo vapors to enter the Engine Room.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Oxygen Analyzer" Calibration
*   **Trade Trick:** Always calibrate the $O_2$ sensor using **Fresh Air (20.9%)** and **Span Gas (Oxygen-free Nitrogen)** before every cargo operation. Sensors "drift" rapidly in the presence of hot combustion gasses.

## 5.2 Detecting a Leaking Deck Water Seal
*   **Trade Trick:** If you suspect vapors are leaking back, check the atmosphere at the IGG blower inlet with a portable gas detector. **If you find Hydrocarbons (HC) there, your Deck Water Seal or Non-Return Valve is failing.**

## 5.3 Clearing a Blocked Demister
*   **Expert Insight:** If the gas pressure is high but flow is low, the "Demister Pad" (wire mesh) at the top of the scrubber is likely "salted up" or clogged with soot. **Trade Trick:** Back-wash the demister with fresh water for 1 hour while the IGG is stopped.

## 5.4 "Ghost" High Temp Alarms
*   **Trade Trick:** If the gas feels cool but the alarm is active, the PT100 sensor in the scrubber outlet is likely coated in sulfuric acid scale. Clean the sensor with a Scotch-Brite pad.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Pre-Operation Checklist
*   **Deck Seal:** Verify water flow and level.
*   **P/V Breaker:** Check the liquid level (usually Glycol/Water mix to prevent freezing).
*   **Analyzers:** Perform the 2-point calibration.

## 6.2 Quarterly Scrubber Inspection
*   **Check:** Open the inspection manhole and check the "Plastic Saddles" (packing). If they are melted or crushed, scrubbing efficiency will drop and gas temperature will rise.

## 6.3 Annual Burner Overhaul
*   **Method:** Replace the fuel nozzle and check the ignition electrodes for "cracks" in the porcelain. A weak spark is the #1 cause of "Failed to Ignite" alarms.

---

# Part 7 – Miscellaneous Knowledge

*   **Topping Up:** When at sea, cargo tanks lose pressure due to cooling (diurnal variation). The IGG is used to "Top Up" the pressure to prevent air from being sucked in via the P/V valves.
*   **Double Block and Bleed:** The IGG system must have two valves and a vent between the machinery space and the deck to ensure absolute isolation when not in use.

**End of Document**
