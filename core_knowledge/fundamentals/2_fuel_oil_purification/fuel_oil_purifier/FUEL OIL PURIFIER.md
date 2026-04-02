# MARINE FUEL OIL PURIFIER – COMPLETE CORE KNOWLEDGE

**Equipment:** Centrifugal Oil Separator (e.g., Alfa Laval S/P Series, GEA Westfalia OSE/OSD, Mitsubishi SJ-H)

**Folder Name:** fuel_oil_purifier

**Prepared by:** Senior Marine Engineer & Separation Technology Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Centrifugal Separation

## 1.1 Stokes' Law and Accelerated Gravity
The separation of water and solids from fuel oil relies on the density difference between the components. In a settling tank, this happens via gravity, but it is too slow for modern high-consumption engines.
*   **The Physics:** $V_g = \frac{d^2 (\rho_p - \rho_f) g}{18 \mu}$
    Where $V_g$ is the settling velocity. To speed this up, the purifier replaces gravity ($g$) with centrifugal force ($r \omega^2$).
*   **G-Force:** A modern purifier rotating at 10,000 RPM can generate forces up to **15,000 G**, reducing separation time from days to seconds.

## 1.2 The Disc Stack Physics (Increasing Surface Area)
The "Disc Stack" is the most important part of the purifier.
*   **Physics:** By dividing the oil into hundreds of thin layers (typically 0.5mm thick), the distance a particle must travel to hit a "collection surface" is reduced. Once a particle hits a disc, it slides outward to the sludge space, preventing it from being re-entrained in the clean oil flow.

## 1.3 Purifier vs. Clarifier (The Interface)
*   **Purifier:** Removes water and solids. It uses a **Water Seal** to create an interface between oil and water.
*   **Clarifier:** Removes only solids and small amounts of water. No water seal is used; it is used as a second stage for extra-dirty fuel.
*   **The Interface Physics:** The position of the "E-line" (interface) is controlled by the **Gravity Disc** (Regulating Disc). If the interface moves too far out, oil overflows through the water outlet. If it moves too far in, water enters the clean oil.

---

# Part 2 – Major Components & Systems

## 2.1 The Bowl Assembly
*   **Bowl Body and Hood:** High-tensile stainless steel designed to withstand massive centrifugal stress.
*   **Sliding Bowl Bottom:** A hydraulic piston that drops momentarily to discharge accumulated sludge (The "Shot").
*   **Disc Stack:** Hundreds of conical stainless steel discs with "caulks" (spacers).

## 2.2 The Drive System
*   **Vertical Shaft:** Carries the bowl. Supported by high-speed precision bearings.
*   **Horizontal Shaft:** Connected to the electric motor via a **Friction Clutch**.
*   **Friction Clutch:** Allows the motor to reach full speed while the heavy bowl gradually accelerates, preventing motor burn-out.

## 2.3 Control System (e.g., EPC-60, CUX)
*   **Operating Water System:** High-pressure water used to open/close the sliding bowl bottom.
*   **Pneumatic/Electric Valves:** Control the "Oil Feed," "Water Seal," and "Discharge" cycles.
*   **Vibration Switch:** Shuts down the unit instantly if imbalance occurs (e.g., uneven sludge buildup).

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Sludge Discharge:** A distinct "thump" followed by a brief rise in motor amperage.
*   **Oil Quality:** Clear, with water content < 0.05% at the outlet.
*   **Vibration:** Minimal (usually < 4.5 mm/s).

## 3.2 Typical Operating Parameters (HFO Separation)

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Separation Temp (HFO)** | 95°C – 98°C | < 90°C (Poor Separation) |
| **Separation Temp (MGO)** | 40°C – 45°C | > 60°C (Fire Risk) |
| **Throughput (Flow)** | 15% – 25% of Capacity | > 50% (High Velocity) |
| **Discharge Interval** | 30 – 120 minutes | > 4 hours (Bowl Clogging) |
| **Water Seal Press** | Stable | Sudden Drop (Overflow) |

## 3.3 The Physics of Temperature
Heating the fuel (to 98°C for HFO) is critical because it reduces the oil's viscosity and increases the density difference between oil and water. **Separation efficiency drops by 50% for every 10°C drop in temperature.**

---

# Part 4 – Common Faults & Root Causes

## 4.1 Overflow (Loss of Water Seal)
*   **Symptom:** Oil coming out of the water outlet; "Low Pressure" alarm on clean oil line.
*   **Root Cause:** Incorrect gravity disc size, sudden change in fuel density, or failed "Operating Water" supply preventing the bowl from staying closed.

## 4.2 High Vibration
*   **Symptom:** Machine shaking; automatic shutdown.
*   **Root Cause:** Broken disc caulk, uneven sludge discharge (partial shot), or worn vertical shaft bearings.
*   **Expert Insight:** If a purifier vibrates after cleaning, you likely put the discs back in the wrong order or left one out.

## 4.3 "Cat Fines" (Catalytic Fines)
*   **Symptom:** Rapid wear of engine fuel pumps and liners.
*   **Physics:** Cat fines are tiny, ultra-hard aluminum/silicon particles from the refinery process. They are slightly denser than oil.
*   **Root Cause:** Operating the purifier at too high a flow rate. To remove cat fines, you must run at the **lowest possible flow rate** to increase "Residence Time" in the bowl.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 Selecting the Correct Gravity Disc (The Nomogram)
Don't guess. Use the manufacturer's Nomogram.
*   **Trade Trick:** You need the **Specific Gravity (Density)** of the oil at 15°C and the actual **Separation Temperature**. If you change bunkers and the new fuel is denser, you MUST change the gravity disc to a smaller inner diameter.

## 5.2 The "Friction Pad" Check
*   **Trade Trick:** Time the start-up. If the bowl takes > 10 minutes to reach full speed, the friction pads are worn or oily. If it takes < 5 minutes, the motor is being overloaded.

## 5.3 Partial vs. Total Discharge
*   **Partial Discharge:** Only sludge is removed (uses less water).
*   **Total Discharge:** The entire bowl content is dumped (cleans the discs better).
*   **Expert Insight:** If the purifier is "ghosting" (overflowing randomly), perform a Total Discharge to clear any hard-packed sludge blocking the water seal paths.

## 5.4 The "Bowl Ring" Tool
*   **Trade Trick:** Never use a hammer and chisel on the bowl lock ring. Always use the official heavy-duty spanner and a "slugging" hammer. Damage to the lock ring can lead to catastrophic bowl failure due to centrifugal stress.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Routine Checks
*   **Operating Water:** Ensure the header tank is full and the filters are clean.
*   **Gearbox Oil:** Change every 6 months (use high-speed gear oil, typically ISO VG 150).

## 6.2 The "Major Overhaul" (Every 8,000 Hours)
*   **Bearings:** Replace all vertical and horizontal shaft bearings.
*   **O-Rings:** Replace every single rubber seal. Use Molykote or silicone grease on the "Sliding Bowl Bottom" seals to ensure they move freely.
*   **Disc Stack:** Clean in a chemical bath (Ultrasound is best) to remove hard carbon.

---

# Part 7 – Miscellaneous Knowledge

*   **ALCAP System (Alfa Laval):** A modern system that doesn't use gravity discs. It uses a "Water Transducer" in the clean oil line to detect water and triggers a discharge automatically.
*   **Sludge Tank Management:** Always monitor the sludge tank level. A blocked sludge pipe will cause the purifier to overflow internally, potentially contaminating the engine room floor.

**End of Document**
