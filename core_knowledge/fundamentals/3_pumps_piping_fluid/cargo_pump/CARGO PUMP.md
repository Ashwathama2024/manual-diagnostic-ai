# MARINE CARGO PUMP – COMPLETE CORE KNOWLEDGE

**Equipment:** High-Capacity Cargo Discharge System (Centrifugal, Deepwell, and Screw types) - (e.g., Framo, Shinko, Hamworthy, Marflex)

**Folder Name:** cargo_pump

**Prepared by:** Senior Marine Engineer & Tanker Operations Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Cargo Discharge

## 1.1 The Physics of NPSH (Net Positive Suction Head)
The performance of a cargo pump is strictly limited by the available pressure at the suction inlet.
*   **The Physics:** $NPSH_a = P_{atm} + P_{static} - P_{vapor} - P_{friction}$
    If $NPSH_a$ falls below the pump's required $NPSH_r$, the cargo will "boil" at the impeller eye, causing **Cavitation**.
*   **Tanker Context:** As the tank level drops, $P_{static}$ decreases. This is why cargo pumps must slow down or use "Stripping" systems as the tank reaches the bottom.

## 1.2 Viscosity and Reynolds Number Physics
*   **The Physics:** High viscosity cargos (like crude oil or molasses) increase the frictional resistance in the piping. 
*   **Centrifugal Pump Limit:** Centrifugal pumps lose efficiency rapidly as viscosity increases. For very thick cargos, **Positive Displacement Screw Pumps** or **Heated Cargo** are required to maintain flow.

## 1.3 Deepwell vs. Pump Room Physics
*   **Pump Room System:** A few massive centrifugal pumps in a dedicated space. Physics relies on high suction lift (limiting for high-vapor-pressure cargos).
*   **Deepwell System (Framo):** Each tank has its own submerged pump. **Physics:** The pump is located at the bottom of the tank, pushing the cargo "up." This eliminates suction lift issues and allows for much more efficient discharge of volatile cargos.

---

# Part 2 – Major Components & System Layout

## 2.1 The Pump Unit (Centrifugal Type)
*   **Impeller:** Designed for high flow (up to 5,000 m³/h). 
*   **Wear Rings:** Maintain the pressure seal between the discharge and suction sides.

## 2.2 Deepwell Hydraulic System (Framo)
*   **HPU (Hydraulic Power Unit):** Central pumps providing 250+ bar to the hydraulic motors on each cargo pump.
*   **The "Power Head":** The submerged hydraulic motor that drives the impeller.

## 2.3 Mechanical Seals and Cofferdams
*   **Mechanical Seal:** Prevents cargo from entering the pump bearings or the environment.
*   **Cofferdam (Framo):** A small air-filled space between the cargo seal and the hydraulic seal. **Physics:** Any leak (cargo or oil) collects here and can be detected via a "Purge Pipe."

## 2.4 Stripping System
A small auxiliary pump or an "Automatic Stripping System" (e.g., VAC-STRIP) that removes air from the main pump suction to allow discharge down to the last few millimeters.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Stability:** Discharge pressure is steady at the manifold.
*   **Sound:** Smooth, low-frequency hum. No "rattling" (cavitation).
*   **Leakage:** Zero cargo visible in the cofferdam purge.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Discharge Pressure** | 7.0 – 12.0 bar | > 14.0 bar (Relief Valve) |
| **Hydraulic Drive Press**| 150 – 250 bar | > 300 bar (System Trip) |
| **Bearing Temp** | 45°C – 65°C | > 85°C (High Temp) |
| **Seal Purge Rate** | < 100 ml / 24h | High Leakage (Seal Fail) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Cavitation (Loss of Suction)
*   **Symptom:** Discharge pressure drops and fluctuates; loud "gravel" noise.
*   **Root Cause:** Pumping too fast at low tank levels, or the cargo is too hot (high vapor pressure).

## 4.2 Mechanical Seal Failure
*   **Symptom:** Cargo detected in the cofferdam purge line.
*   **Root Cause:** Abrasive particles in the cargo (sand/rust) grinding the seal faces, or "Dry Running" the pump.

## 4.3 Hydraulic Motor Failure (Framo)
*   **Symptom:** Pump will not rotate even with full hydraulic pressure.
*   **Root Cause:** Internal "Vane" or "Piston" damage in the motor due to contaminated hydraulic oil.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Purging" Routine (Framo)
*   **Trade Trick:** Always purge the cofferdam **before and after** every cargo operation. 
    *   **Cargo Leak:** Seal is failing. 
    *   **Hydraulic Oil Leak:** The hydraulic motor seal is failing. 
    *   **Expert Insight:** A small amount of "condensate" (water) is normal; don't panic unless it's pure cargo.

## 5.2 The "Back-Pressure" Trick for Cavitation
*   **Trade Trick:** If the pump starts to cavitate, **slightly close the discharge valve** on the pump. This increases the internal pressure in the pump casing and can "collapse" the cavitation bubbles, allowing you to continue pumping at a lower rate.

## 5.3 Detecting a "Blocked Suction"
*   **Expert Insight:** If the pump discharge pressure is high but the flow at the manifold is zero, check the **Cargo Strainer** in the tank. If a plastic liner or rag is sucked in, it will act like a "Flap Valve."

## 5.4 Starting a "Stuck" Pump
*   **Trade Trick:** If a hydraulic pump has been sitting for months and won't start, increase the hydraulic pressure to maximum and **momentarily reverse the flow** (if the system allows) to "break" the static friction on the seals.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Routine Checks
*   **Hydraulic Oil:** Maintain ultra-clean oil (**ISO 16/14/11**). 90% of cargo pump failures are caused by dirty hydraulic oil.
*   **Wear Rings:** Check the clearance during drydock. If it exceeds 2mm, the "Internal Recycling" will reduce pump capacity by 20%.

## 6.2 Mechanical Seal Overhaul
*   **Critical Action:** When replacing a seal, never touch the silicon carbide faces with your bare fingers. The oils from your skin can cause the seal to "hot spot" and fail.

## 6.3 Pressure Testing the Cargo Main
*   **Method:** Perform a hydrostatic test of the deck lines every 12 months at 1.5x working pressure to ensure no "Thinning" from corrosion has occurred.

---

# Part 7 – Miscellaneous Knowledge

*   **IGS (Inert Gas System):** Cargo pumps must never be used unless the tank is "Inerted" (Oxygen < 8%). The friction from a failing bearing can easily provide the ignition source for a tank explosion.
*   **COW (Crude Oil Washing):** Some cargo is diverted back into the tank via "COW Machines" to wash down the walls and "liquefy" the sludge at the bottom for more efficient discharge.

**End of Document**
