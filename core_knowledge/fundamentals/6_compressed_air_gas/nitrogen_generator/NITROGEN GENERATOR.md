# MARINE NITROGEN GENERATOR – COMPLETE CORE KNOWLEDGE

**Equipment:** Nitrogen Production System (Membrane or PSA type) - (e.g., Air Products, Parker, Generon, Atlas Copco)

**Folder Name:** nitrogen_generator

**Prepared by:** Senior Marine Engineer & Specialized Cargo Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Gas Separation

## 1.1 The Physics of Selective Permeation (Membrane Type)
Most modern Nitrogen ($N_2$) generators on ships use Hollow Fiber Membranes.
*   **The Physics:** Air is composed of approx. 78% Nitrogen and 21% Oxygen. The membrane fibers are made of specialized polymers that allow "Fast" gasses (Oxygen, $CO_2$, and Water Vapor) to permeate through the fiber walls while keeping the "Slow" gas (Nitrogen) inside the fiber.
*   **Driving Force:** The separation relies on the **Partial Pressure Gradient** across the membrane. High-pressure feed air (approx. 10–13 bar) is required to push the unwanted gasses through the fiber walls.

## 1.2 Pressure Swing Adsorption (PSA) Physics
*   **The Physics:** PSA uses "Carbon Molecular Sieves" (CMS). At high pressure, Oxygen molecules are trapped (adsorbed) in the pores of the CMS, while Nitrogen molecules pass through. 
*   **The Cycle:** When the CMS becomes saturated with Oxygen, the pressure is released (vented), and the Oxygen is released to the atmosphere, "regenerating" the sieve.

## 1.3 Purity vs. Flow Physics
*   **The Physics:** There is an inverse relationship between Nitrogen purity and flow rate. 
*   **Trade-off:** If you increase the flow rate, the air has less "residence time" in the membrane, and purity drops (e.g., from 99.9% to 95%). For chemical tankers, 99.9% purity is often required to prevent cargo oxidation.

---

# Part 2 – Major Components & System Layout

## 2.1 Feed Air Compressor
Usually a dedicated high-pressure screw compressor. It must provide large volumes of air at constant pressure.

## 2.2 Pre-Treatment Module (Crucial)
*   **Physics:** The membranes are extremely sensitive to liquid water and oil. 
*   **Components:** Coalescing filters, activated carbon towers (to remove oil vapor), and a **Refrigerated Dryer**. **Requirement:** Air must be "Instrument Quality" before it hits the membranes.

## 2.3 Air Heater
*   **Function:** Heats the feed air to approx. 40°C – 50°C. 
*   **Physics:** Permeability increases with temperature. Heating the air makes the membrane more efficient and prevents any remaining moisture from condensing inside the fibers.

## 2.4 The Membrane Bundle (The Heart)
Thousands of hair-like hollow fibers packed into a stainless steel pressure vessel.

## 2.5 Oxygen Analyzer and Off-Spec Valve
*   **Analyzer:** Monitors the purity of the produced Nitrogen.
*   **Off-Spec Valve:** If purity drops below the setpoint (e.g., > 1% $O_2$), this 3-way valve automatically diverts the "dirty" gas to the atmosphere rather than the cargo tanks.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Purity:** Stable at the commanded setpoint (e.g., 99% for general inerting, 99.9% for cargo padding).
*   **Sound:** A steady hum from the compressor and a faint "hiss" from the membrane permeate vent.
*   **Filters:** Differential pressure gauges are in the green zone.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Feed Air Pressure** | 10.0 – 13.0 bar | < 9.0 bar (Low Flow) |
| **Feed Air Temp** | 45°C – 55°C | > 65°C (Membrane Risk) |
| **Nitrogen Purity** | 95% – 99.9% | < Setpoint (Off-Spec) |
| **Permeate Flow** | Consistent | Sudden Drop (Membrane Fail) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Oil Contamination (The Membrane Killer)
*   **Symptom:** Nitrogen purity drops rapidly even with high air pressure.
*   **Root Cause:** Failed oil separator in the feed compressor or saturated carbon filters.
*   **Physics:** Oil coats the surface of the fibers, blocking the pores. **Once a membrane is oil-fouled, it is destroyed and must be replaced.**

## 4.2 Low Purity (High $O_2$)
*   **Symptom:** $O_2$ analyzer reading rises above 1%.
*   **Root Cause:** Feed air temperature too low or flow rate too high for the desired purity.
*   **Expert Insight:** Often caused by a leaking "Product Valve" that is allowing more gas out than the membrane can handle.

## 4.3 High Feed Air Temperature
*   **Symptom:** "High Air Temp" trip.
*   **Root Cause:** Failed air heater thermostat or restricted cooling air on the compressor.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Zirconia Sensor" Calibration
*   **Trade Trick:** Most $N_2$ systems use Zirconia Oxygen sensors. These sensors require high heat to work. **Always allow 30 minutes of warm-up time** before trusting the purity reading. Calibrate with 99.99% pure Nitrogen (Span Gas) annually.

## 5.2 Detecting a Ruptured Fiber
*   **Trade Trick:** If you hear a loud "Whistle" from inside the membrane housing, a fiber has likely snapped. This will cause an immediate and massive drop in Nitrogen purity.

## 5.3 The "Carbon Filter" Smell Test
*   **Expert Insight:** Open the drain on the final pre-filter before the membrane. **If you can smell oil, your carbon tower is saturated.** Stop the system immediately to save the membranes (each bundle can cost $10,000+).

## 5.4 Optimizing Purity vs. Fuel
*   **Trade Trick:** If you only need 95% Nitrogen for basic inerting, **reduce the heater setpoint or increase the flow**. This makes the compressor work less, saving fuel and wear. Only run at 99.9% when the cargo requires it.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Filter Management (Critical)
*   **Weekly:** Check the manual drains on the pre-treatment filters.
*   **Quarterly:** Replace the coalescing filter elements.
*   **Annual:** Replace the Activated Carbon tower filling.

## 6.2 Membrane Efficiency Test
*   **Method:** Once a year, record the exact Feed Air Pressure, Temperature, and Flow required to reach 99.0% purity. Compare this to the factory commissioning data. **A 10% increase in required pressure indicates the membranes are starting to foul.**

## 6.3 Screw Compressor Service
*   **Check:** Nitrogen compressors run at higher pressures than working air compressors. Use "Semi-Synthetic" or "Full Synthetic" oil to prevent oil breakdown and carry-over.

---

# Part 7 – Miscellaneous Knowledge

*   **Nitrogen Padding:** Maintaining a positive pressure of $N_2$ in a chemical tank to prevent moisture or air from entering.
*   **Cargo Stripping:** Nitrogen is often used to "blow" the remaining cargo out of the lines into the shore manifold, ensuring zero cargo loss.

**End of Document**
