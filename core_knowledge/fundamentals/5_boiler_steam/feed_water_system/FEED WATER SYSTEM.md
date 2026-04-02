# MARINE BOILER FEED WATER SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Boiler Feed Water Supply and Control System (Pumps, Hotwell, and De-aeration)

**Folder Name:** feed_water_system

**Prepared by:** Senior Marine Engineer & Steam Plant Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Water Supply

## 1.1 The Physics of Pumping into Pressure
The feed water pump must provide enough pressure to overcome the boiler's internal steam pressure plus the height (Head) of the boiler.
*   **The Physics:** $P_{pump} > P_{boiler} + \rho g h + P_{friction}$
    If the boiler is at 7 bar, the pump must typically produce **10–12 bar** to ensure a steady flow into the drum.

## 1.2 The Physics of "Steam Binding" (Cavitation)
Feed water is often kept very hot (80°C – 95°C) in the hotwell to prevent thermal shock.
*   **The Problem:** At these temperatures, the water is close to its boiling point. If the pressure at the pump suction drops slightly, the water "flashes" into steam.
*   **Result:** The pump becomes "Steam Bound." It cannot pump gas, the flow stops, and the boiler water level drops instantly.

## 1.3 De-aeration Physics (Oxygen Removal)
*   **The Physics:** The solubility of oxygen in water decreases as the temperature increases. 
*   **The Process:** By heating the feed water to 95°C in the **De-aerator / Hotwell**, dissolved oxygen is forced out of the water. This is critical because oxygen causes "Pitting Corrosion" which can eat through a boiler tube in weeks.

---

# Part 2 – Major Components & System Layout

## 2.1 The Hotwell (Observation Tank)
*   **Function:** Collects the condensate returning from the ship's heaters. It is also the "Filter Tank" where any oil contamination is detected.
*   **Layout:** Divided into compartments with "Weirs" and Loofah sponges to trap oil.

## 2.2 Feed Water Pumps
Usually multi-stage centrifugal pumps.
*   **Configuration:** At least two pumps (Duty/Standby). They are high-precision units with very tight internal clearances.

## 2.3 Feed Water Control Valve
A modulating valve (pneumatic or electric) controlled by the boiler's level sensor.
*   **Physics:** It uses a "Three-Element Control" on large boilers (Level + Steam Flow + Feed Flow) to anticipate load changes.

## 2.4 Feed Water Heater / Economizer
A heat exchanger that uses exhaust gas or steam to give the water a final "pre-heat" before it enters the boiler drum.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Level Control:** The valve moves smoothly; the boiler water level is stable within ±50mm.
*   **Pump Sound:** A consistent, high-pitched "Whine." No "Cracking" sounds (cavitation).
*   **Transparency:** The water in the observation tank is clear and free of oil "sheen."

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Hotwell Temperature** | 85°C – 95°C | < 80°C (Oxygen Risk) |
| **Pump Discharge Press** | 10 – 15 bar | < 8 bar (Low Flow) |
| **Dissolved Oxygen** | < 0.02 mg/l | > 0.1 mg/l (Corrosion) |
| **Filter ΔP** | 0.2 bar | > 0.5 bar (Clean Filters) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Steam Binding (The "Hot Pump" Fault)
*   **Symptom:** Pump is very hot to the touch; discharge pressure drops to zero; loud rattling.
*   **Root Cause:** Hotwell temperature too high (boiling) or the suction filter is blocked.
*   **Physics:** The vacuum created by the pump suction lowers the boiling point of the already hot water.

## 4.2 Oil Contamination
*   **Symptom:** Oil visible in the observation tank or the "Oil-in-Water" alarm triggers.
*   **Root Cause:** A leak in a fuel oil heater or a lube oil heater.
*   **Danger:** Oil in the feed water will "coat" the boiler tubes, preventing heat transfer and causing them to overheat and explode.

## 4.3 Level Control "Hunting"
*   **Symptom:** The feed valve cycles from 0% to 100% repeatedly; water level is unstable.
*   **Root Cause:** Improperly tuned PID controller or air in the pneumatic control lines.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 Curing "Steam Binding" with Cold Water
*   **Trade Trick:** If a pump is steam bound, **DO NOT pour cold water on the outside of the pump** (it will crack). Instead, spray the **suction pipe** with a cold-water hose to condense the steam bubbles inside, or briefly add a small amount of cold make-up water to the hotwell.

## 5.2 The "Recirculation Valve" Check
*   **Expert Insight:** Feed pumps have a "Minimum Flow" or "Recirculation" line. If the boiler is at full level and the valve closes, the pump will overheat and seize. **Ensure the recirculation line is always warm**, proving that some water is flowing back to the hotwell.

## 5.3 Manual Level Control (The "Two-Man" Routine)
*   **Trade Trick:** If the automatic controller fails, one person must stay at the **Gauge Glass** and radio the level to the second person at the **Manual Bypass Valve**. **Small adjustments only!** It takes 30 seconds for a valve change to show in the glass.

## 5.4 Detecting "Air Leaks" via the Sight Glass
*   **Trade Trick:** If you see tiny air bubbles in the boiler sight glass, air is being sucked in on the **Suction Side** of the feed pump. Check the pump gland packing immediately.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Pump Swap:** Switch between the duty and standby feed pumps.
*   **Sight Glass Cleaning:** Clean the observation tank sight glass to ensure oil is visible.

## 6.2 Cleaning the Observation Tank
*   **Method:** Every 3 months, drain the hotwell and replace the loofah sponges or "Filter Coir." Wash out any accumulated sludge from the bottom.

## 6.3 Pump Overhaul
*   **Check:** The "Impeller Clearances." Due to the high pressure, even 0.5mm of wear will significantly reduce the pump's "Dead-Head" pressure, making it unable to feed the boiler at full steam pressure.

---

# Part 7 – Miscellaneous Knowledge

*   **Make-up Water:** Fresh water from the FWG is used to replace water lost during blow-down. This water is "Dead" (no oxygen) but can be slightly acidic.
*   **Hydro-test Pump:** Most feed systems have a small hand-pump used for "Hydrostatic Testing" the boiler after repairs (testing to 1.5x working pressure).

**End of Document**
