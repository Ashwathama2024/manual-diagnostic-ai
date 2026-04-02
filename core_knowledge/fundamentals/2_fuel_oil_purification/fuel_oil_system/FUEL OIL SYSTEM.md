# MARINE FUEL OIL SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Fuel Supply and Circulation System (Low-Pressure & High-Pressure Circuits)

**Folder Name:** fuel_oil_system

**Prepared by:** Senior Marine Engineer & Fuel Systems Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Fuel Handling

## 1.1 The Physics of Atomization (Viscosity Control)
For efficient combustion in a diesel engine, fuel must be "atomized" (broken into tiny droplets) by the fuel injectors.
*   **The Physics:** Atomization is highly dependent on the **Kinematic Viscosity** $(\nu)$ of the fuel at the moment of injection. 
*   **Viscosity Target:** For Heavy Fuel Oil (HFO), the target viscosity is typically **10–15 cSt**. 
*   **Temperature Dependence:** HFO has a high viscosity at room temperature (up to 700 cSt). To reach 12 cSt, it must be heated to approximately **135°C–150°C**.

## 1.2 Pressure Circuit Physics
The system is divided into two loops:
1.  **Feed (Supply) Loop:** Low pressure (approx. 4 bar). Moves fuel from the service tank to the mixing column.
2.  **Circulation (Booster) Loop:** Medium pressure (approx. 7–10 bar). Circulates heated fuel through the engine's fuel rail to prevent "waxing" or cooling. 
*   **Physics of Pressure:** The booster loop pressure must be kept high enough to prevent the fuel from "gassing" or "boiling" at 150°C.

## 1.3 The Mixing Column (Degassing) Physics
As fuel is heated and circulated, trapped air and light-end gasses (vapors) are released.
*   **The Component:** The Mixing Column (Venting Tank) provides a space for these vapors to separate and be vented to the service tank, preventing "Vapor Lock" in the fuel pumps.

---

# Part 2 – Major Components & System Layout

## 2.1 Settling and Service Tanks
*   **Settling Tank:** Where fuel is held to allow water and heavy particles to settle before purification.
*   **Service Tank:** Holds clean, purified fuel ready for the engine.
*   **Heating Coils:** Use steam or thermal oil to maintain fuel temperature above its "Pour Point."

## 2.2 Feed and Circulation Pumps
*   **Feed Pumps:** Positive displacement pumps (usually gear or screw type) that move fuel to the booster module.
*   **Circulation (Booster) Pumps:** High-capacity pumps that circulate 3–4 times more fuel than the engine consumes to ensure stable temperature in the fuel rail.

## 2.3 Fuel Heaters and Viscosity Controller
*   **Viscometer:** A sensitive instrument that measures the oil's actual viscosity and sends a signal to the steam control valve.
*   **Steam/Electric Heaters:** Provide the final heat boost to reach the 12 cSt target.

## 2.4 Automatic Backwash Filter
*   **Component:** A 10–50 micron filter that automatically cleans itself using a small amount of "back-flush" oil when the differential pressure ($\Delta P$) rises.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Viscosity:** Steady at the setpoint (e.g., 12.5 cSt).
*   **Temperature:** Stable and correlated with the fuel's viscosity curve.
*   **Leakage:** No leakage from the shielded high-pressure pipes.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Fuel Viscosity (at engine)** | 10 – 15 cSt | < 8 cSt (Leak) / > 20 cSt (Smoke) |
| **Fuel Temp (at engine)** | 135°C – 150°C | > 155°C (Gassing) |
| **Booster Loop Pressure** | 7.0 – 10.0 bar | < 5 bar (Vapor Lock) |
| **Feed Loop Pressure** | 3.5 – 5.0 bar | < 2.5 bar (Cavitation) |
| **Filter ΔP** | 0.2 – 0.4 bar | > 0.8 bar (Clogged) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Vapor Lock (Gassing)
*   **Symptom:** Engine loses power or "hunts" (RPM fluctuates); circulation pumps make a "gravel" sound (cavitation).
*   **Root Cause:** Fuel is too hot or booster pressure is too low.
*   **Physics:** The fuel's vapor pressure exceeds the line pressure, creating gas bubbles that the pumps cannot move.

## 4.2 Thermal Shock (The Changeover Fault)
*   **Symptom:** Fuel pump plungers "seize" during changeover from HFO to MGO.
*   **Root Cause:** Temperature change too rapid (> 2°C per minute).
*   **Physics:** The plunger and barrel have different masses and expansion coefficients. A sudden blast of cold MGO shrinks the barrel before the plunger, causing them to lock.

## 4.3 High Differential Pressure ($\Delta P$)
*   **Symptom:** "Auto-Filter Fault" alarm.
*   **Root Cause:** "Cat Fines" or wax buildup in the fuel, or a failure of the filter's air-driven cleaning motor.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "HFO to MGO" Changeover (The 2°C/Min Rule)
*   **Trade Trick:** Never just flip the valve. You must manually ramp down the heater setpoint while slowly introducing MDO into the mixing column. **The goal is a slow, steady blend.** 
*   **Physics:** Monitor the "Viscosity Curve" on the HMI. If the curve is a straight line, you are changing too fast.

## 5.2 Emergency Fuel Cut-Off (Quick-Closing Valves)
*   **Critical Safety:** In case of an engine room fire, the **Quick-Closing Valves (QCV)** on the settling/service tanks can be tripped remotely (usually by air or wire). **Always test the QCVs during drydock or major port stays.**

## 5.3 Detecting a "Leaking" Injector via Pressure
*   **Trade Trick:** If the fuel rail pressure drops rapidly after stopping the circulation pumps, one or more fuel injectors is likely "leaking" into the cylinder, posing a risk of **Hydraulic Lock** on start-up.

## 5.4 Viscometer Cleaning
*   **Expert Insight:** If the viscosity reading is "frozen" or wildly inaccurate, the viscometer's sensing capillary is likely blocked with carbon. **Do not adjust the heater manually for long; clean the sensor immediately.**

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Routine
*   **Drainage:** Drain water and sludge from the settling and service tanks every morning.
*   **Pressure Check:** Verify the pressure drop across the "Safety Filter" (the last manual filter before the engine).

## 6.2 Pump Overhaul
*   **Mechanical Seals:** Fuel pumps use carbon-face mechanical seals. If you see a "drip" from the pump tell-tale hole, the seal is failing and needs replacement before it sprays hot oil.

## 6.3 Heater Cleaning (Descaling)
*   **Method:** Every 1-2 years, the HFO heaters will become "coked" with baked oil. They must be cleaned with specialized chemicals (Carbon Remover) to maintain heat transfer efficiency.

---

# Part 7 – Miscellaneous Knowledge

*   **LSFO (Low Sulphur Fuel Oil):** Modern VLSFO (Very Low Sulphur) is often unstable. Mixing two different VLSFOs in the same tank can cause "Incompatibility," leading to massive sludge precipitation and blocked filters. **Always keep different bunker batches in separate tanks.**
*   **Fuel Meters:** Most modern ships use **Coriolis Flow Meters**, which measure Mass Flow ($kg/h$) directly, rather than volume, providing much more accurate SFOC (Specific Fuel Oil Consumption) data.

**End of Document**
