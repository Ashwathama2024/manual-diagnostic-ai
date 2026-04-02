# MARINE CENTRAL COOLING SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Integrated Central Freshwater Cooling System (PHEs, LT/HT Loops, and Temperature Control)

**Folder Name:** central_cooling_system

**Prepared by:** Senior Marine Engineer & Thermodynamic Systems Expert (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Central Cooling

## 1.1 The Physics of Secondary Heat Exchange
Most modern ships use a "Central" system to protect machinery from the corrosive and abrasive effects of raw seawater.
*   **The Physics:** Heat is moved in two stages:
    1.  **Stage 1:** Individual components (Lube oil, air, jackets) reject heat into a clean Freshwater loop.
    2.  **Stage 2:** The Freshwater loop rejects the accumulated heat into Seawater via large **Central Heat Exchangers**.
*   **Physics of Protection:** Freshwater is a closed loop. Seawater only enters the Engine Room at a single point (the Central PHE), minimizing the risk of leaks and internal corrosion.

## 1.2 The Two-Loop Architecture (HT and LT)
*   **HT (High Temperature) Loop:** $80^\circ C – 95^\circ C$. Focuses on the engine cylinder jackets. **Physics:** Prevents "Cold Corrosion" and maintains thermal expansion of the liners.
*   **LT (Low Temperature) Loop:** $35^\circ C – 45^\circ C$. Focuses on the Lube Oil, Charge Air, and Auxiliary equipment. **Physics:** The LT loop acts as the "Cold Reservoir" for the entire ship.

## 1.3 Total Heat Load Physics
*   **The Physics:** $Q_{total} = Q_{ME} + Q_{Gen} + Q_{AC} + Q_{Aux}$. 
*   **Energy Balance:** The central cooling system must be sized to reject the **Peak Heat Load** occurring during tropical navigation with the engine at 100% MCR.

---

# Part 2 – Major Components & System Layout

## 2.1 Central Plate Heat Exchangers (PHE)
The "Barrier" between the ship and the ocean.
*   **Material:** Usually Titanium plates to resist the combined attack of seawater and high velocity.
*   **Physics:** Uses "Counter-flow" arrangement to maximize the Logarithmic Mean Temperature Difference ($\Delta T_{lm}$), achieving 95%+ efficiency.

## 2.2 Circulating Pump Banks
*   **SW Pumps:** Move seawater from the sea chests through the PHEs.
*   **LT Pumps:** Move freshwater from the PHEs to the machinery.
*   **Redundancy:** Usually 2 Duty + 1 Standby arrangement.

## 2.3 De-aeration and Expansion Tank
A single expansion tank usually serves both loops. **Physics:** Provides static head to the pumps and allows for the removal of "Entrained Air" which would cause cavitation and reduced heat transfer.

## 2.4 Control Valves (The "Mixers")
Pneumatic or electric 3-way valves that determine how much LT water is bypassed around the PHE to maintain a constant $38^\circ C$ supply temperature.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **LT Inlet Temp:** Rock-steady at $38^\circ C$.
*   **Pressure Drop:** $\Delta P$ across the PHEs matches the nameplate (e.g., 0.5 bar).
*   **Expansion Tank:** Level is stable; no "Bubbling" (which would indicate an internal engine leak).

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **LT Supply Temp** | 36°C – 40°C | > 42°C (Cooling Limit) |
| **HT Supply Temp** | 80°C – 85°C | < 75°C (Corrosion Risk) |
| **PHE SW ΔP** | 0.4 – 0.7 bar | > 1.0 bar (Silt/Shells) |
| **PHE FW ΔP** | 0.2 – 0.4 bar | > 0.6 bar (Oil/Scaling) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Cross-Contamination" (Plate Leak)
*   **Symptom:** Freshwater expansion tank level rises; chloride levels increase in the FW.
*   **Root Cause:** Pinhole leak in a titanium plate due to erosion or "Plates Chattering."
*   **Physics:** If the seawater pump pressure exceeds the freshwater pressure, salt-water enters the clean loop.

## 4.2 High LT Temperature (Tropical Limit)
*   **Symptom:** LO and Charge air temperatures rise even with 3 SW pumps running.
*   **Root Cause:** Heavy bio-fouling on the seawater side of the central PHE.
*   **Solution:** Check the MGPS (Chlorination) system and perform a "Back-flush."

## 4.3 Control Loop "Oscillation"
*   **Symptom:** Supply temperature swings from $30^\circ C$ to $45^\circ C$.
*   **Root Cause:** Improper PID tuning or air in the pneumatic control actuator.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Touch and Go" PHE Check
*   **Trade Trick:** Touch the four corners of the central PHE. 
    *   **Hot In / Cold Out (FW side):** Good heat rejection.
    *   **Cold In / Hot Out (SW side):** Good heat absorption.
    *   **Expert Insight:** If the SW outlet is cold, but the FW outlet is hot, the water is "Short-circuiting" inside the PHE due to a damaged internal gasket.

## 5.2 Finding a Leaking Plate via "Isolation"
*   **Trade Trick:** If you have high chlorides, isolate one PHE at a time. **Method:** Close the SW valves and vent the SW side. If the FW level stops rising, the isolated PHE is the one with the leak.

## 5.3 The "Air Pocket" Purge
*   **Expert Insight:** After cleaning a PHE, air often traps in the top of the plates. **Trade Trick:** Keep the top vent valve open while slowly filling the unit until a solid stream of water comes out. **Physics:** Even 10% air-volume in a PHE can reduce cooling capacity by 50%.

## 5.4 Using the "Emergency Cross-over"
*   **Trade Trick:** If the LT pumps fail, you can sometimes use the **Fire Pump** to feed the LT loop via a jumper hose (if fitted). **Warning:** This is for "Get-home" only; the water will be saltwater, and you must drain and chemically flush the entire system immediately after.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Chemical Analysis
*   **Nitrite Level:** Must be 1500–3000 ppm. **Physics:** Protects the engine liners from "Micro-pitting."
*   **Chloride Level:** Must be < 50 ppm. **Physics:** Prevents stress-corrosion cracking of stainless steel components.

## 6.2 Annual PHE Service
*   **Method:** Disassemble the PHE. Clean plates with a soft brush. **Inspection:** Look for "Plate Thinning" near the ports. **Tightening:** Always use a calibrated torque wrench to tighten the plate pack to the exact "A-dimension" stamped on the frame.

## 6.3 Pump Seal Replacement
*   **Method:** LT pumps run continuously. Replace mechanical seals proactively every 2 years.

---

# Part 7 – Miscellaneous Knowledge

*   **Jacket Water Pre-heater:** Often integrated into the central system. It uses steam or electricity to keep the HT water at $60^\circ C$ while the engine is stopped.
*   **Heat Recovery:** Waste heat from the HT loop is diverted to the **Freshwater Generator** to produce drinking water.

**End of Document**
