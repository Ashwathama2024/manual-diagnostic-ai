# MARINE SCRUBBER SYSTEM (EGCS) – COMPLETE CORE KNOWLEDGE

**Equipment:** Exhaust Gas Cleaning System (e.g., Wärtsilä, Alfa Laval PureSOx, Clean Marine, Yara Marine)

**Folder Name:** scrubber_system

**Prepared by:** Senior Marine Engineer & Environmental Systems specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of SOx Removal

## 1.1 The Chemistry of Sulfur Neutralization
The primary goal of the scrubber is to remove Sulfur Dioxide ($SO_2$) from the exhaust gasses.
*   **The Reaction:** $SO_2 + H_2O \rightarrow H_2SO_3$ (Sulfurous Acid).
*   **Neutralization Physics:** The acidic wash-water is neutralized by the natural alkalinity of seawater ($HCO_3^-$ bicarbonate ions). 
    \[ H_2SO_3 + 2HCO_3^- \rightarrow SO_4^{2-} + 2H_2O + 2CO_2 \]
    The sulfur is converted into harmless **Sulfate** ($SO_4$), which is already a natural component of seawater.

## 1.2 Open-Loop vs. Closed-Loop Physics
1.  **Open-Loop:** Uses raw seawater. **Physics:** Relies on the massive volume and natural alkalinity of the ocean. Most efficient in high-salinity deep sea.
2.  **Closed-Loop:** Recirculates fresh water with a chemical buffer (Sodium Hydroxide - $NaOH$). **Physics:** Used in low-salinity areas (Baltic Sea) or "Zero Discharge" zones.
3.  **Hybrid:** Can switch between both modes.

## 1.3 Gas-Liquid Contact Physics
*   **The Physics:** To remove 98%+ of $SO_x$, every molecule of gas must contact a droplet of water. 
*   **Method:** Uses "Spray Headers" and "Packed Beds" (random or structured packing) to maximize the **Surface Area** ($A$) and **Residence Time**. 
*   **Back-Pressure Physics:** The scrubber adds resistance to the exhaust flow. If the pressure drop is too high, the engine's turbocharger efficiency will drop, and scavenge air pressure will fall.

---

# Part 2 – Major Components & System Layout

## 2.1 The Scrubber Tower
A large stainless steel (SMO 254 or Alloy 59) vessel located inside the funnel.
*   **Material Physics:** Must resist extreme heat ($400^\circ C$) when dry and extreme acid ($pH 2.0$) when running.

## 2.2 Wash-Water Pumps
High-capacity centrifugal pumps.
*   **Redundancy:** Essential for compliance. If the pumps stop, the ship must switch to expensive Low-Sulfur fuel (MGO) immediately.

## 2.3 CEMS (Continuous Emissions Monitoring System)
*   **Function:** Measures the ratio of $SO_2$ (ppm) to $CO_2$ (%). 
*   **The Limit:** To be equivalent to 0.1% sulfur fuel, the ratio must be below **4.3**.

## 2.4 Water Quality Monitoring (WQM)
Measures the discharge water for **pH**, **PAH** (Polycyclic Aromatic Hydrocarbons), and **Turbidity**.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Emissions:** $SO_2/CO_2$ ratio stable at 2.0 – 4.0.
*   **Plume:** A white "Steam Plume" from the funnel is normal (condensing water vapor). 
*   **Water Discharge:** $pH$ is neutralized back to $> 6.5$ before it leaves the ship's side (or 4 meters from the shell).

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **SO2 / CO2 Ratio** | 2.5 – 4.0 | > 4.3 (Non-Compliance) |
| **Exhaust ΔP** | 100 – 300 mmWG | > 500 mmWG (Engine Load Red.) |
| **Wash-water Flow** | 500 – 1500 m³/h | Low Flow (Acid Risk) |
| **Discharge pH** | 6.5 – 7.5 | < 6.5 (Illegal Discharge) |
| **Discharge Turbidity**| < 20 FNU | > 25 FNU (Soot Carry-over) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 CEMS Analyzer Drift (False Alarms)
*   **Symptom:** Ratio climbs to 10+ while the engine is healthy.
*   **Root Cause:** Water droplets in the sample line or a fouled optical window.
*   **Expert Insight:** CEMS systems are notoriously finicky. 90% of "Exceedance" events are due to sensor failure, not scrubbing failure.

## 4.2 Corrosion of the Scrubber Body
*   **Symptom:** Liquid leaking from the funnel base.
*   **Root Cause:** "Acid Concentration." 
*   **Physics:** If the wash-water doesn't fully cover the walls, "dry spots" form where the acid becomes highly concentrated and eats through even specialized stainless steel.

## 4.3 High Exhaust Back-Pressure
*   **Symptom:** Engine "Surging"; high exhaust temperatures.
*   **Root Cause:** "Packed Bed" is clogged with soot or the "Demister" is fouled.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Sample Line" Blow-back
*   **Trade Trick:** If the $SO_2$ reading is "frozen," the sample line is likely blocked with condensate. Use the **manual blow-back** function with dry instrument air for 5 minutes. If that fails, the "Heated Hose" has a failed heating element.

## 5.2 Detecting a "Dry Run" Accident
*   **Expert Insight:** If the scrubber was accidentally operated without water for even 10 minutes, the internal plastic nozzles may have melted. **Trade Trick:** Perform a "Nozzle Check" by running the pumps while the engine is stopped and looking through the inspection port for a uniform spray pattern.

## 5.3 The "Alkalinity Trick" for Open-Loop
*   **Trade Trick:** If you are in a low-alkalinity area (e.g., near a large river mouth) and the ratio is borderline, **increase the wash-water flow to maximum.** This provides more "Bicarbonate" molecules to the reaction zone even if their concentration is low.

## 5.4 WQM Sensor Cleaning
*   **Trade Trick:** The discharge water sensors (PAH/Turbidity) get coated in "Bio-film" and soot. **Clean them weekly with a soft toothbrush and vinegar.** Do not use detergents, as they will trigger a false PAH alarm.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Compliance Log
*   **Requirement:** The "EGCS Record Book" must be kept up to date. You must record any "Exceedance" over 4.3 and the action taken (e.g., "Changed to MGO").

## 6.2 Quarterly Nozzle Inspection
*   **Check:** Remove and inspect the spray nozzles for erosion. If the orifice is enlarged, the droplet size will increase, and **Scrubbing Efficiency will drop by 20%.**

## 6.3 Annual CEMS Calibration
*   **Requirement:** The gas analyzers must be calibrated using "Certified Span Gasses" (e.g., 100ppm $SO_2$ in $N_2$) every 12 months.

---

# Part 7 – Miscellaneous Knowledge

*   **VGP (Vessel General Permit):** In US waters, scrubber discharge is governed by strict VGP rules. You must have a "VGP Test" performed by a laboratory annually.
*   **Zero-Discharge Mode:** In some ports (like Singapore), open-loop scrubbing is banned. You must either switch to **Closed-Loop (Holding Tank)** or switch to **Low-Sulfur Fuel**.

**End of Document**
