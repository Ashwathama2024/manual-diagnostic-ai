# MARINE STEAM TURBINE (AUXILIARY) – COMPLETE CORE KNOWLEDGE

**Equipment:** Auxiliary Steam Turbines (e.g., Shinko, Mitsubishi, Dresser-Rand, Elliott) for Cargo Pumps and Turbogenerators.

**Folder Name:** steam_turbines

**Prepared by:** Senior Marine Engineer & Steam Systems Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Steam Expansion

## 1.1 The Physics of Energy Conversion
A steam turbine converts the thermal energy (Enthalpy) of high-pressure steam into mechanical rotational energy.
*   **The Physics:** $V = \sqrt{2 \cdot \Delta H}$
    As steam expands through a nozzle, its pressure drops and its velocity $(V)$ increases. This high-speed jet hits the turbine blades, transferring its momentum to the rotor.
*   **Impulse vs. Reaction:** 
    *   **Impulse Physics:** The pressure drop occurs entirely in the stationary nozzles. The blades are "pushed" by the high-velocity jet.
    *   **Reaction Physics:** The pressure drops across both the stationary nozzles and the moving blades. The blades move due to the "Kickback" (thrust) of the steam.

## 1.2 The Physics of Critical Speed and Vibration
Steam turbines rotate at very high speeds (3,000 to 10,000 RPM).
*   **The Physics:** Every rotor has a **Natural Frequency**. If the turbine rotates at this speed (Critical Speed), vibration will increase exponentially, leading to catastrophic failure (Blades hitting the casing).
*   **Governing:** The control system must ensure the turbine passes through its critical speed range quickly and never operates continuously within it.

## 1.3 Condensing vs. Back-Pressure Physics
*   **Condensing Turbine:** Exhausts into a vacuum ($0.1$ bar). This maximizes the "Heat Drop" $(\Delta H)$, providing high efficiency.
*   **Back-Pressure Turbine:** Exhausts into a low-pressure steam line (e.g., 2 bar). Efficiency is lower, but the exhaust steam can be used for heating other ship systems.

---

# Part 2 – Major Components & System Layout

## 2.1 The Rotor and Blading
*   **Rotor:** A high-tensile steel shaft carrying the blade wheels.
*   **Blades (Buckets):** Aerodynamically shaped to capture steam energy. Made of specialized stainless steel or nickel alloys to resist **Erosion** and **Creep**.

## 2.2 The Governor (Speed Control)
*   **Type:** Hydraulic (Woodward) or Electronic.
*   **Function:** Controls the "Governor Valve" to maintain constant RPM regardless of the load (e.g., as a cargo pump's suction changes).

## 2.3 Gland Sealing System
*   **The Problem:** At the high-pressure end, steam wants to leak out. At the vacuum end, air wants to leak in.
*   **The Physics Solution:** "Labyrinth Seals" and **Gland Steam**. A small amount of low-pressure steam (0.1 bar) is fed into the seals to create a "Steam Barrier" that prevents air ingress and steam leakage.

## 2.4 Emergency Trip Valve (Stop Valve)
A fast-acting valve that slams shut in < 0.5 seconds if a safety limit is exceeded. It is independent of the governor valve.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Sound:** A high-pitched, smooth "Whine." Any "Rumbling" or "Vibration" is a major warning.
*   **Vibration:** Levels should be < 2.5 mm/s (RMS) at the bearings.
*   **Oil:** Clear and bright. No foaming.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Turbine Speed** | 3600 / 6000 RPM | > 110% (Overspeed Trip) |
| **LO Inlet Temp** | 40°C – 45°C | > 55°C (Bearing Heat) |
| **Exhaust Vacuum** | 650 – 720 mmHg | < 500 mmHg (Trip) |
| **Gland Steam Press**| 0.1 – 0.2 bar | Low (Air Leak) / High (Leak) |
| **Vibration** | < 2.5 mm/s | > 7.0 mm/s (Emergency Stop) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Overspeed (The #1 Hazard)
*   **Symptom:** Rapid rise in RPM; high-pitched scream.
*   **Root Cause:** Sudden loss of load (e.g., pump cavitation) combined with a "Sticky" governor valve.
*   **Physics:** Centrifugal force increases with the square of speed ($RPM^2$). At 120% speed, the blades can physically detach from the rotor.

## 4.2 Water Carry-over (Slugging)
*   **Symptom:** Sudden "Banging" sound; massive vibration; drop in RPM.
*   **Root Cause:** Boiler priming (high water level/TDS).
*   **Physics:** Water is much denser than steam. Hitting a blade at 300 m/s with a "Slug" of water is like hitting it with a brick. It will snap blades instantly.

## 4.3 Loss of Vacuum
*   **Symptom:** Exhaust temperature rises; turbine slows down.
*   **Root Cause:** Air leak in the gland seals or a failed condensate pump.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Slow Roll" Warm-up
*   **Trade Trick:** Never start a turbine from cold. Use the "Bypass" valve to rotate the turbine at **200–500 RPM for 20 minutes**. 
*   **Physics:** This ensures the rotor heats up evenly. If you heat one side only, the rotor will "Bow" (The Hogging effect), leading to massive vibration on start-up.

## 5.2 The "Hand-Trip" Test
*   **Trade Trick:** Every time you start the turbine, manually trigger the **Emergency Trip** while it is at low speed. If the valve doesn't slam shut instantly, **Shut down the steam supply manually and do not use the turbine.** A sticky trip valve is a "Bomb" waiting to happen.

## 5.3 Checking "Carbon Rings"
*   **Expert Insight:** Carbon gland rings are brittle. If you see steam leaking from the shaft ends even with gland steam on, the rings are likely cracked. **Trade Trick:** You can sometimes "Heal" a minor leak by temporarily increasing the gland steam pressure, but this will contaminate the Lube Oil with water over time.

## 5.4 The "Feel" for Bearing Wear
*   **Trade Trick:** Place your hand on the bearing housing. A "Smooth Hum" is good. A "Gritty Vibration" indicates the white metal is starting to flake (Spalling).

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Routine
*   **Oil Level:** Check the sump level and look for water (milky color).
*   **Filter Check:** Clean the LO duplex filters. Check for shiny metallic particles.

## 6.2 Annual Governor Service
*   **Method:** Flush the hydraulic oil in the governor. **Calibrate the speed setpoints** using a strobe-tachometer.

## 6.3 Borescope Inspection (Internal)
*   **Check:** Every 2 years, remove a nozzle block or inspection plug and borescope the "First Stage" blades. Look for **Erosion** at the tips and **Scaling** (Salt buildup) which can unbalance the rotor.

---

# Part 7 – Miscellaneous Knowledge

*   **Sentinel Valve:** A small relief valve on the turbine casing. It doesn't protect the turbine from overpressure, but it **Whistles** to warn the engineer that the exhaust pressure is too high.
*   **Turning Gear:** Large turbines use a motor-driven turning gear to keep the rotor spinning after shutdown. This prevents the shaft from bending while it cools down.

**End of Document**
