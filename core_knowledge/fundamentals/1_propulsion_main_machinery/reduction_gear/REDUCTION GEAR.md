# MARINE REDUCTION GEAR – COMPLETE CORE KNOWLEDGE

**Equipment:** Marine Reduction Gearbox (e.g., RENK, Reintjes, ZF, MAN, Wärtsilä)

**Folder Name:** reduction_gear

**Prepared by:** Expert Marine Mechanical Engineer & Gear Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Gear Reduction

## 1.1 The Physics of Torque Multiplication
Marine engines (especially medium and high-speed types) rotate much faster than an efficient propeller. The reduction gear matches these speeds.
*   **Gear Ratio ($i$):** $i = n_{in} / n_{out} = z_{out} / z_{in}$
    Where $n$ is speed (RPM) and $z$ is the number of teeth.
*   **Torque Multiplication:** As speed decreases, torque increases proportionally (ignoring efficiency losses).
    \[ Q_{out} = Q_{in} \cdot i \cdot \eta \]
    This allows a high-speed engine to turn a large-diameter propeller with massive rotational force.

## 1.2 Tooth Contact Physics (Hertzian Stress)
The transfer of power occurs across a tiny line of contact on the gear teeth.
*   **Surface Stress:** Known as Hertzian Contact Stress. If the oil film breaks down, the stress exceeds the material's yield point, leading to "Pitting."
*   **Helical vs. Spur Gears:** Marine gears almost always use **Helical Gears**. The teeth are cut at an angle, allowing multiple teeth to be in contact at once. This distributes the load more smoothly and reduces noise/vibration.

## 1.3 The Physics of the Clutch
Most gearboxes include a clutch (hydraulic or pneumatic) to disconnect the engine from the shaft.
*   **Friction Physics:** The clutch relies on the friction coefficient ($\mu$) between steel and friction plates. If the clamping pressure drops, the plates slip, generating intense heat that can warp the plates in seconds.

---

# Part 2 – Major Components & Systems

## 2.1 Pinions and Bull Gear
*   **Pinion:** The smaller, high-speed gear connected to the engine. It is the most highly stressed component.
*   **Bull Gear:** The large, low-speed gear connected to the propeller shaft.
*   **Material:** Usually high-alloy case-hardened steel (e.g., 18CrNiMo7-6), precision ground to DIN 4 or 5 accuracy.

## 2.2 Bearings and Thrust Internal
*   **Journal Bearings:** White-metal lined bearings that support the shafts.
*   **Integrated Thrust Bearing:** Many gearboxes house the ship's main thrust bearing at the aft end of the bull gear shaft to absorb propeller thrust.

## 2.3 Lubrication System (The Lifeblood)
*   **Main Oil Pump:** Often driven directly by the gearbox input shaft.
*   **Spray Nozzles:** These inject a "curtain" of oil directly into the "mesh" (where the teeth meet).
*   **Oil Cooler:** Critical for maintaining viscosity. High oil temp = thin oil = metal-to-metal contact.

## 2.4 The Clutch (Multi-Disc)
*   **Hydraulic Actuation:** Uses oil pressure (typically 20–30 bar) to compress the friction discs.
*   **Engagement Control:** A "Proportional Valve" ensures the clutch engages slowly to prevent a "shock load" that could snap a shaft.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Noise:** A consistent "whine" is normal. A "growl" or "clunk" is not.
*   **Oil Color:** Clear and amber. Black or "glittery" oil indicates a major failure.
*   **Tooth Surface:** A polished "mirror" finish on the load side of the teeth.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Lubrication Oil Pressure** | 2.5 – 5.0 bar | < 1.8 bar (Shutdown) |
| **Clutch Engagement Press** | 20 – 28 bar | < 15 bar (Slippage Risk) |
| **Oil Temp (Inlet)** | 40°C – 50°C | > 65°C (High Temp) |
| **Bearing Temp** | 50°C – 70°C | > 85°C (Failure) |
| **Filter Differential Press** | < 0.5 bar | > 1.2 bar (Clogged) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Pitting (Surface Fatigue)
*   **Symptom:** Small craters on the tooth surface.
*   **Root Cause:** Overloading or poor oil quality. The cyclic stress causes micro-cracks that eventually "pop" out a piece of metal.

## 4.2 Scuffing / Scoring
*   **Symptom:** Vertical scratches or "smearing" of metal on the teeth.
*   **Root Cause:** Instantaneous breakdown of the oil film due to high heat or "Cold Start" loading.

## 4.3 Clutch Slippage
*   **Symptom:** Propeller RPM is lower than the calculated gear ratio suggests; oil temp rises rapidly.
*   **Root Cause:** Leaking internal seals in the clutch piston or worn friction plates.

## 4.4 Gear Misalignment
*   **Symptom:** Uneven wear pattern on the teeth (e.g., wear only on the aft side of the gear).
*   **Root Cause:** Hull deflection or worn journal bearings allowing the shaft to tilt.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Blueing Test" (Contact Pattern)
How do you check if the teeth are aligned?
*   **Method:** Apply a thin layer of "Engineers Blue" (Prussian Blue) to three teeth on the pinion. Rotate the gears by hand. The pattern transferred to the bull gear tells you the alignment.
*   **Healthy:** Pattern should cover > 80% of the tooth width.

## 5.2 Checking "Backlash"
Backlash is the tiny gap between teeth to allow for thermal expansion and lubrication.
*   **Method:** Lock the bull gear and move the pinion back and forth with a dial gauge.
*   **Expert Insight:** Too little backlash = Gear seizure when hot. Too much backlash = Hammering and noise.

## 5.3 Borescope Inspection
*   **Trade Trick:** You can inspect 90% of the gear teeth through the inspection covers using a high-definition borescope without draining the oil. Look specifically at the "Pitch Line" for the first signs of pitting.

## 5.4 The "Magnet Check"
*   **Trade Trick:** Most gearbox filters have magnetic plugs. Check them weekly. 
    *   **Fine Grey Sludge:** Normal wear.
    *   **Shiny Shards:** Tooth spalling (Pitting).
    *   **Yellow/Bronze Flakes:** Bearing or thrust pad distress.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Checks
*   **Oil Level:** Check while the gearbox is running (if wet sump).
*   **Spray Pattern:** Observe through the sight glass that oil is spraying across the full width of the gears.

## 6.2 Oil Analysis (Quarterly)
The most important preventive tool.
*   **Iron (Fe):** Indicates gear wear.
*   **Copper (Cu):** Indicates thrust pad or clutch plate wear.
*   **Viscosity:** Must remain within 10% of the original (typically ISO VG 150 or 220).

## 6.3 Filter Cleaning
When cleaning the duplex filter, always inspect the "safety mesh" for any large debris that could indicate a sudden mechanical failure.

---

# Part 7 – Miscellaneous Knowledge

*   **PTO / PTI (Power Take-Off / Power Take-In):** Many gearboxes have an extra shaft to drive a generator (PTO) or to allow an electric motor to drive the propeller (PTI/Hybrid).
*   **Turning Gear:** Always engage the turning gear before a "Cold Start" to ensure an oil film is present on the teeth before they take full engine torque.

**End of Document**
