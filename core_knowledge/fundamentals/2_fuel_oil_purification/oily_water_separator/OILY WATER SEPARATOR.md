# MARINE OILY WATER SEPARATOR (OWS) – COMPLETE CORE KNOWLEDGE

**Equipment:** Bilge Water Separator (e.g., Victor Marine, RWO, Alfa Laval BlueBox, JFE)

**Folder Name:** oily_water_separator

**Prepared by:** Expert Marine Environmental & Systems Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Oil Separation

## 1.1 MARPOL Annex I – The Regulatory Physics
The OWS is the primary tool for complying with MARPOL Annex I. 
*   **The Physics of Compliance:** It is strictly prohibited to discharge bilge water with an oil content greater than **15 parts per million (PPM)**. 
*   **The 3-Way Valve:** If the oil content exceeds 15 PPM, the "fail-safe" physics of the system automatically diverts the water back to the bilge tank or the oily water holding tank.

## 1.2 Stokes' Law and Gravitational Separation
Like the purifier, the OWS uses density differences, but it usually operates under static gravity rather than centrifugal force.
*   **The Physics:** $V = \frac{g \cdot D^2 (\rho_w - \rho_o)}{18 \mu}$
    Because the velocity $(V)$ of a tiny oil droplet rising in water is very slow, the OWS must use **Coalescence** to speed it up.

## 1.3 Coalescence Physics (Increasing Droplet Size)
*   **Physics:** $D$ (droplet diameter) is the most powerful variable in Stokes' Law ($D^2$). 
*   **The Process:** The bilge water passes through a "Coalescer" (a filter-like mesh or plate stack). Small oil droplets collide with the mesh and merge into larger droplets. Once the droplet is large enough, its buoyancy increases rapidly, and it floats to the top of the separator.

---

# Part 2 – Major Components & System Layout

## 2.1 Primary Separation Chamber
The first stage where heavy oil and solids settle out. It usually contains a baffle or a "corrugated plate interceptor" (CPI).

## 2.2 Coalescer Chamber (Second Stage)
Houses the coalescer filters or membranes. This stage handles the "fine" separation down to the 15 PPM limit.

## 2.3 Oil Content Monitor (OCM)
The "Watchman" of the system.
*   **Physics:** It uses **Light Scattering** technology. A beam of light passes through the water. If oil is present, the light is scattered or absorbed. Sensors measure the intensity of the light to calculate the PPM.

## 2.4 The 15 PPM Stop / 3-Way Valve
An automatic solenoid-controlled valve. 
*   **De-energized State (Fail-Safe):** Recirculates back to the tank.
*   **Energized State:** Discharges overboard (only if < 15 PPM).

## 2.5 The Bilge Pump
Usually a positive displacement pump (like a mono-pump or diaphragm pump) to avoid "emulsifying" the oil before it reaches the separator. High-speed centrifugal pumps are avoided.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **OCM Reading:** Stable between 0 and 5 PPM.
*   **Oil Discharge:** Occurs periodically when the oil level probe in the top of the separator detects an oil layer.
*   **Clarity:** The water sample at the "Test Cock" should be clear and odorless.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Oil Content** | 0 – 5 PPM | > 15 PPM (Recirculate) |
| **Operating Pressure**| 1.0 – 2.0 bar | High Press (Filter Clog) |
| **Flow Rate** | Within Rated Capacity | > Rated (Poor Separation) |
| **OCM Status** | "Sample Flow OK" | "No Sample" (System Stop) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "OCM High Alarm" (Constant Recirculation)
*   **Symptom:** The OWS never discharges overboard; reading stays at 15+ PPM.
*   **Root Cause:** Chemical Emulsion.
*   **Physics:** Detergents (from deck cleaning) lower the surface tension of the oil, creating "Chemical Emulsions" that are too small to coalesce. No gravity-based OWS can separate a chemical emulsion.

## 4.2 Fouled OCM Glass
*   **Symptom:** Reading stays at 0 or is wildly unstable.
*   **Root Cause:** Biofilm or oil coating on the internal measuring glass.
*   **Expert Insight:** Even a fingerprint on the glass can cause a 10 PPM error.

## 4.3 High Differential Pressure
*   **Symptom:** Bilge pump works harder; flow decreases.
*   **Root Cause:** Coalescer filters are blinded by "Black Sludge" (soot and wax from fuel/LO leakage).

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Fresh Water Flush" Test
*   **Trade Trick:** To verify if the OCM is faulty or if the water is actually dirty, switch the OCM inlet to "Fresh Water." The reading should drop to 0 or 1 PPM instantly. If it stays at 15+, the OCM is fouled or faulty.

## 5.2 Cleaning the OCM Sample Line
*   **Trade Trick:** If the OCM is erratic, use the manufacturer's cleaning brush (with a small amount of mild detergent) to clean the internal tube. **Never use abrasive cleaners.**

## 5.3 The "Air Vent" Routine
*   **Expert Insight:** Air bubbles in the sample line look like oil droplets to the light sensors, triggering a false 15 PPM alarm. Always ensure the "Air Vent" on top of the OWS is open until a solid stream of water comes out before starting the OCM.

## 5.4 Detecting "Illegal" Detergents
*   **Expert Insight:** If the bilge water is "cloudy" but the OWS is healthy, check the engine room chemicals. Only use "Quick-Break" detergents that are designed to allow oil and water to separate in the bilge.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Verification
*   **Test Discharge:** Manually trigger the 3-way valve to ensure it moves freely.
*   **Sample Verification:** Take a sample and compare it to the OCM reading.

## 6.2 Filter Replacement
*   **Method:** Always change the primary stage filter before it reaches the "High Diff Press" alarm. Once a coalescer is blinded, it cannot be cleaned; it must be replaced.

## 6.3 OCM Calibration (Every 5 Years)
*   **Requirement:** The OCM must be calibrated by an authorized service provider. Check the **Certificate of Type Approval** and ensure the seal on the OCM is intact.

## 6.4 Oil Record Book (ORB)
*   **Critical Action:** Every operation of the OWS must be recorded in the ORB Part 1. Ensure the "Stop Time," "Start Time," and "Total Volume Discharged" match the **VDR (Voyage Data Recorder)** and the ship's GPS coordinates.

---

# Part 7 – Miscellaneous Knowledge

*   **White Box / Blue Box:** Some OWS systems are integrated into a "Tamper-Proof" box that records all data to prevent illegal bypasses.
*   **Oil Discharge Monitoring (ODME):** Similar to OWS but used for "Cargo Slops" on tankers. It uses a different limit (30 Liters per Nautical Mile).

**End of Document**
