# MARINE STEERING GEAR – COMPLETE CORE KNOWLEDGE

**Equipment:** Hydraulic Steering Gear (e.g., Rapson Slide / Ram Type, Rotary Vane) - (e.g., Rolls-Royce, Hatlapa, Kawasaki, Porsgrunn)

**Folder Name:** steering_gear

**Prepared by:** Senior Marine Engineer & Hydraulics Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Steering

## 1.1 Pascal's Law and Hydraulic Force
The steering gear relies on Pascal’s Law: "Pressure applied to an enclosed fluid is transmitted undiminished to every portion of the fluid and to the walls of the containing vessel."
*   **Physics:** $F = P \cdot A$
    A small amount of pressure $(P)$ from the hydraulic pump acts on a large piston area $(A)$ of the steering ram to create the massive force $(F)$ required to turn the rudder against the sea flow.

## 1.2 Rapson Slide Physics (Torque Variation)
In a ram-type steering gear, the torque exerted on the rudder stock varies with the angle.
*   **The Physics:** Torque $Q = F \cdot L \cdot \cos(\theta)$. 
    As the rudder turns to high angles, the effective lever arm changes. The Rapson Slide mechanism is designed to compensate for this, providing increasing mechanical advantage as the rudder reaches high angles where water resistance is greatest.

## 1.3 Rudder Torque Requirements (SOLAS)
SOLAS (Safety of Life at Sea) mandates specific performance physics:
*   **Main Steering:** Must be able to move the rudder from 35° on one side to 35° on the other side at maximum ahead speed.
*   **The 28-Second Rule:** The movement from 35° to 30° on the opposite side must take no more than 28 seconds. This requires a high-volume, variable-displacement hydraulic pump.

---

# Part 2 – Major Components & Systems

## 2.1 The Actuators (The Muscles)
*   **Ram Type:** Two or four hydraulic cylinders acting on a tiller arm. Highly robust and easy to repair.
*   **Rotary Vane Type:** A compact unit where hydraulic pressure acts directly on vanes attached to the rudder stock. Takes up less space but has complex internal seals.

## 2.2 The Pumps (The Heart)
*   **Variable Displacement Pumps:** (e.g., Hele-Shaw or Axial Piston). These pumps can change the volume and direction of oil flow without changing motor speed. 
*   **The Hunting Gear (Feedback):** A mechanical linkage (or electronic equivalent) that "shuts off" the pump flow once the rudder reaches the commanded angle.

## 2.3 The Tiller and Rudder Stock
*   **Tiller:** The "lever" that turns the rudder.
*   **Rudder Stock:** The massive vertical shaft that holds the rudder. It is supported by a **Carrier Bearing** which takes the entire weight of the rudder assembly.

## 2.4 Control Systems (The Brain)
*   **Telemotor:** The system that transmits the bridge steering wheel command to the steering gear room. Can be hydraulic (older ships) or electric/digital (modern).
*   **Autopilot:** Interfaces with the telemotor to maintain a constant heading.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Movement:** Smooth, quiet, and without "shuddering."
*   **Response:** The rudder follows the wheel exactly with no "lag" or "overshoot."
*   **Redundancy:** Both pump units (System 1 and System 2) are ready for immediate start.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **System Pressure** | 50 – 150 bar | > 250 bar (Relief Valve) |
| **Header Tank Level** | 60% – 80% | Low Level (Leakage) |
| **Oil Temperature** | 35°C – 50°C | > 65°C (Cooler Fail) |
| **Rudder Response** | < 1 second lag | High Deadband |
| **Motor Current** | Stable during swing | High Amps (Mechanical Bind) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Air in the System (The "Spongy" Feel)
*   **Symptom:** Rudder movement is jerky; "milky" oil in the tank; audible "hissing" or "cracking" in the pipes.
*   **Root Cause:** Low oil level in the header tank allowing air to be sucked into the pump suction.

## 4.2 Hydraulic Lock
*   **Symptom:** The pump motor is running, but the rudder will not move in one or both directions.
*   **Root Cause:** A failed non-return valve or a blocked pilot-operated check valve.

## 4.3 Rudder Hunting
*   **Symptom:** The rudder constantly moves back and forth (±1°) around the setpoint.
*   **Root Cause:** Wear in the feedback linkage (Hunting Gear) or air in the telemotor system.

## 4.4 Carrier Bearing Wear (Jumping Rudder)
*   **Symptom:** Excessive vibration and "thumping" when the ship is in heavy seas.
*   **Root Cause:** The thrust washer in the carrier bearing has worn down, allowing the rudder stock to move vertically.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 Emergency Steering (The "Trick Wheel")
Every engineer must know how to steer from the Steering Gear Room.
*   **Method:** Switch the telemotor to "Local." Use the small handwheel (Trick Wheel) on the pump to manually control the stroke.
*   **Trade Trick:** If the electronics fail, you can manually push the solenoid valves on the pump to move the rudder.

## 5.2 The "Bypass Test" for Internal Leaks
How do you know if the cylinder seals are leaking?
*   **Method:** Put the rudder hard-over to 35° and close the isolating valves. If the rudder "creeps" back towards the center, the internal piston seals or the relief valves are leaking.

## 5.3 Bleeding Air "On the Fly"
*   **Trade Trick:** If air enters the system while at sea, turn the rudder hard-over to hard-over multiple times. This forces the air into the cylinders where it can be bled off via the "Air Vent" screws at the highest point of the system.

## 5.4 Checking "Jumping Stopper" Clearance
*   **Trade Trick:** There is a "Stopper" above the rudder stock to prevent the rudder from jumping out in heavy seas. Check the clearance (usually 2-5mm). If it's zero, the rudder carrier bearing has failed and is being crushed.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Checks
*   **Oil Level:** Check the header tanks.
*   **Greasing:** Ensure the rudder stock carrier bearing and the Rapson slide pins are well-greased.

## 6.2 Pre-Departure Testing (Mandatory)
Before every port departure, the steering gear must be tested:
1.  Full swing 35° to 35° on Pump 1.
2.  Full swing 35° to 35° on Pump 2.
3.  Test of the **Low Level Alarm**.
4.  Verification of the **Bridge Indicator** vs. the actual rudder position.

## 6.3 Oil Analysis
Hydraulic oil (ISO VG 46 or 68) must be kept ultra-clean. Any grit will destroy the high-precision variable displacement pumps.

---

# Part 7 – Miscellaneous Knowledge

*   **Emergency Power:** The steering gear must be connected to the **Emergency Switchboard** and must be able to restart automatically after a blackout.
*   **Double Acting:** Most systems are designed so that if one pipe bursts, the system can be isolated to run on the remaining cylinders at reduced capacity.

**End of Document**
