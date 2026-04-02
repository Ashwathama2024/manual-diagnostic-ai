# MARINE STERN THRUSTER – COMPLETE CORE KNOWLEDGE

**Equipment:** Aft Transverse Tunnel Thruster (e.g., Kongsberg/Rolls-Royce, Brunvoll, Schottel, Wärtsilä)

**Folder Name:** stern_thruster

**Prepared by:** Senior Marine Engineer & Propulsion Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Stern Thrust

## 1.1 Hydrodynamics of Aft Lateral Force
Like the bow thruster, the stern thruster provides transverse force, but its location at the aft of the vessel changes the maneuverability physics.
*   **Physics of Rotation:** The stern thruster acts on a longer lever arm relative to the ship’s pivot point (which typically moves forward as the ship gains speed). When used in conjunction with the bow thruster, it allows the ship to "crab" (move sideways) or rotate 360° on its own axis.
*   **Propeller Interaction:** The stern thruster tunnel is often located near the main propeller(s). The "Physics of Interference" means that the main propeller wash can reduce the efficiency of the stern thruster if they are used simultaneously at high power.

## 1.2 Thrust and Power Physics
\[ T = \dot{m} \Delta v \]
In the stern, the tunnel is often shorter or shaped differently to accommodate the hull's "run" toward the transom. This can lead to slightly higher flow losses compared to the bow thruster.

---

# Part 2 – Major Components & Systems

## 2.1 Tunnel Design and Location
*   **Location:** Placed as far aft as possible for maximum lever arm, but deep enough to avoid "air drawing" (ventilation) during pitching in heavy seas.
*   **Grids:** Protect the propeller from debris. In the stern, these must be robust enough to withstand the high-velocity discharge from the main propeller.

## 2.2 Prime Mover and Drive Train
*   **Electric Motor:** Usually a synchronous or induction AC motor. Often located in a dedicated "Aft Thruster Room."
*   **Right-Angle Drive:** A submerged gearbox inside the tunnel, identical in principle to the bow thruster, using spiral bevel gears to transmit power to the propeller.

## 2.3 Pitch Control (CPP)
Most stern thrusters are Controllable Pitch (CPP).
*   **Hydraulic Actuation:** A hydraulic piston inside the hub changes the blade angle.
*   **Zero-Pitch Interlock:** The motor cannot be started unless the blades are in the neutral (zero thrust) position to prevent high-torque motor burn-out.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Vibration:** Low-frequency vibration is normal; high-frequency "grinding" indicates gear distress.
*   **Amperage:** Motor current increases linearly with pitch. At zero pitch, the current should be 20–30% of Full Load Amps (FLA) due to internal friction and water "churning."

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Seal Oil Header Tank** | 60% – 80% Full | Low Level (Leak) |
| **Lube Oil Temp** | 40°C – 60°C | > 75°C (High Temp) |
| **Motor Current (Full Pitch)**| 90% – 95% FLA | > 100% (Trip) |
| **Hydraulic Pressure** | 25 – 45 bar | Low Press (No Pitch) |
| **Insulation Resistance** | > 100 MΩ | < 1 MΩ (Motor Fault) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Ventilation (Air Drawing)
*   **Symptom:** Sudden "thumping" noise and rapid fluctuations in motor amperage.
*   **Root Cause:** The ship is in ballast (light) condition, and the tunnel is too close to the surface, sucking in air.
*   **Physics:** Air has 1/800th the density of water; when the propeller hits air, the load disappears instantly, causing the motor to surge.

## 4.2 Seal Failure (Water Ingress)
*   **Symptom:** Milky oil in the header tank or rising oil level.
*   **Root Cause:** Entangled fishing lines or "ghost nets" in the aft area, which is a high-risk zone for debris.

## 4.3 Bearing Spalling
*   **Symptom:** Fine metallic "glitter" in the gear oil.
*   **Root Cause:** Prolonged operation at high load or water contamination reducing the oil's film strength.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 Manual Pitch Adjustment
*   **Trade Trick:** If the bridge control fails, use the local HPU override solenoids. **Always have a person with a radio at the motor ammeter** to ensure you don't overload the motor while manually "jogging" the pitch.

## 5.2 Detecting Blade Damage via Vibration
*   **Trade Trick:** If one blade is bent, the vibration will be at a frequency of **1x Shaft RPM**. If the gearbox is failing, the vibration will be at the **Tooth Meshing Frequency** (RPM x number of teeth).

## 5.3 Seal Pressure "Trick"
*   **Trade Trick:** If the aft seal is leaking slightly into the sea, you can temporarily lower the header tank level to reduce the static pressure, provided it remains higher than the sea-head. This minimizes environmental impact until repairs.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Checks
*   **Oil Samples:** Check the visual clarity of the gear oil and hydraulic oil daily.
*   **Space Inspection:** Ensure the Aft Thruster Room bilge is dry.

## 6.2 Oil Analysis (Quarterly)
Crucial for detecting gear wear (Iron) and bearing wear (Copper/Lead).
*   **Viscosity:** Monitor for "thickening," which indicates oil oxidation due to overheating.

## 6.3 Diver Inspection (Annual)
*   **Blade Edges:** Check for cavitation erosion.
*   **Zinc Anodes:** Stern thrusters are in a highly galvanic area (near the main propeller and rudder). Anodes often consume faster here and may need replacement every 12 months.

---

# Part 7 – Miscellaneous Knowledge

*   **Dynamic Positioning (DP):** For DP vessels, the stern thruster is a "Critical Component." A failure can lead to a "Loss of Position" alarm.
*   **Anti-Fouling:** The tunnel is a dark, high-flow area where barnacles love to grow. If the tunnel is fouled, the thruster can lose up to 20% of its efficiency.

**End of Document**
