# MARINE EXHAUST GAS ECONOMIZER (EGE) – COMPLETE CORE KNOWLEDGE

**Equipment:** Main Engine Exhaust Gas Boiler / Economizer (e.g., Aalborg, Green's, Mitsubishi, KangRim)

**Folder Name:** exhaust_gas_economizer

**Prepared by:** Senior Marine Engineer & Waste Heat Recovery Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Waste Heat Recovery

## 1.1 The Physics of Convective Heat Transfer
The EGE captures heat from the Main Engine (ME) exhaust gasses ($300^\circ C – 400^\circ C$) that would otherwise be lost up the funnel.
*   **The Physics:** $Q = U \cdot A \cdot \Delta T_{lm}$
    Where $Q$ is the heat recovered, $U$ is the heat transfer coefficient, $A$ is the surface area of the tubes, and $\Delta T_{lm}$ is the Logarithmic Mean Temperature Difference between the gas and the water.
*   **Surface Area Enhancement:** To maximize $A$ in a small space, many EGEs use **Finned Tubes** (Extended Surface). The fins increase the area contact with the exhaust gas by up to 10x.

## 1.2 Soot as a Thermal Insulator
*   **The Physics:** Soot (unburned carbon) is a highly efficient insulator. A layer of soot only **1mm thick** can reduce the heat transfer coefficient ($U$) by **10–15%**, leading to lower steam production and higher exhaust gas back-pressure on the engine.

## 1.3 The Physics of a Soot Fire (The Iron-Steam Reaction)
*   **The Danger:** If soot builds up and the engine is operated at high load, the soot can ignite.
*   **The Physics:** A soot fire can reach temperatures over $1000^\circ C$. If water is sprayed onto the white-hot iron tubes, a catastrophic chemical reaction occurs:
    \[ 3Fe + 4H_2O \rightarrow Fe_3O_4 + 4H_2 \]
    The water decomposes into **Hydrogen Gas**, which explodes, and the iron itself burns as fuel. This is the "Boiler Meltdown" scenario.

---

# Part 2 – Major Components & System Layout

## 2.1 The Tube Bank
*   **Forced Circulation:** In most modern ships, the EGE does not have its own steam drum. Water is pumped from the Auxiliary Boiler's steam drum to the EGE and back.
*   **Risers and Downcomers:** The pipes connecting the EGE to the Aux Boiler.

## 2.2 Circulation Pumps
Dedicated centrifugal pumps that move water through the EGE tubes. **Redundancy:** Always two pumps (Duty/Standby).

## 2.3 Soot Blowers
*   **Method:** Uses high-pressure steam or compressed air nozzles that rotate between the tube banks to "blast" the soot off the surfaces.
*   **Sonic Soot Blowers:** Use high-intensity sound waves to vibrate the soot particles loose.

## 2.4 Exhaust Gas Bypass Damper
A heavy-duty flap valve that can divert the exhaust gas away from the EGE tubes if steam production is not required or if the EGE is being cleaned.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Steam Pressure:** Controlled by the Aux Boiler system (e.g., 7.0 bar).
*   **Gas Differential Pressure ($\Delta P$):** Stable across the tube bank (e.g., 100 – 200 mmWG).
*   **Water Flow:** Constant circulation (proven by the pump discharge pressure).

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Exhaust Gas Inlet** | 250°C – 350°C | Load Dependent |
| **Exhaust Gas Outlet** | 160°C – 200°C | < 140°C (Dew Point Risk) |
| **Gas ΔP** | 150 mmWG | > 300 mmWG (Soot Clog) |
| **Circulation Press** | 2.0 bar above Boiler Press | < 0.5 bar (Low Flow) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Dew Point" Corrosion
*   **Symptom:** Rapid thinning and failure of the lower tube banks.
*   **Root Cause:** Operating the EGE at too low a load or with cold water.
*   **Physics:** If the tube surface temperature drops below the **Acid Dew Point** (approx. $135^\circ C$), sulfuric acid ($H_2SO_4$) condenses from the exhaust gas and eats the steel.

## 4.2 Circulation Pump Failure
*   **Symptom:** "Low Flow" alarm; steam pressure drops.
*   **Root Cause:** Mechanical seal failure or electrical fault.
*   **Danger:** If flow stops while the engine is running, the water in the tubes will evaporate into "Dry Steam," and the tubes will overheat and sag.

## 4.3 Tube Leakage
*   **Symptom:** Auxiliary Boiler water level drops rapidly; white "smoke" (steam) from the funnel.
*   **Root Cause:** Thermal fatigue or oxygen pitting.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Soot Blowing" Routine
*   **Trade Trick:** Always increase the Main Engine load slightly before soot blowing. This increases the gas velocity, which helps carry the loosened soot out of the funnel rather than letting it settle back on the tubes.

## 5.2 Detecting a "Minor" Leak via the Drain
*   **Trade Trick:** Every watch, open the "Exhaust Gas Box Drain" valve at the bottom of the EGE. **If water comes out instead of dry soot, you have a leaking tube.** Catching it early prevents a massive soot fire.

## 5.3 The "Infrared Funnel" Check
*   **Expert Insight:** Use an IR thermometer to check the funnel temperature. If the EGE outlet is significantly hotter than normal for the current engine load, the tubes are **heavily fouled with soot** and acting as an insulator.

## 5.4 Manual Circulation Pump "Swap"
*   **Trade Trick:** Swap the EGE circulation pumps every week. If the standby pump sits for months in hot water without running, the mechanical seal will "bond" to the shaft and fail the moment you need it.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Fire-Side Water Washing
*   **Method:** During port stays, the EGE is washed with high-pressure fresh water. 
*   **Caution:** Ensure the "Brine Drain" is open and the water is neutralized before discharge. The wash-water is highly acidic (pH 2-3).

## 6.2 Soot Blower Inspection
*   **Check:** Ensure the nozzles are not "blowing" directly against the tubes. If a nozzle is misaligned, the high-pressure steam will "erode" the steel tube in a matter of hours (Steam Cutting).

## 6.3 Tube Thickness Gauging (UT)
*   **Check:** During drydock, perform Ultrasonic Thickness (UT) gauging on the "Leading Edge" tubes, as these are the most exposed to erosion and corrosion.

---

# Part 7 – Miscellaneous Knowledge

*   **Turbogenerator:** On very large ships (VLCCs, Container ships), the EGE produces enough steam to drive a **Steam Turbogenerator**, providing "Free" electricity for the ship while at sea.
*   **Cold Start:** Never start the Main Engine without the EGE circulation pumps running. The "Thermal Shock" of $300^\circ C$ gas hitting cold, dry tubes will cause immediate cracking.

**End of Document**
