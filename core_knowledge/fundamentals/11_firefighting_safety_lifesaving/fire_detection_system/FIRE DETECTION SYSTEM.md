# MARINE FIRE DETECTION SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Addressable Marine Fire Detection and Alarm System (e.g., Autronica, Consilium, Tyco, Minerva)

**Folder Name:** fire_detection_system

**Prepared by:** Expert Marine Electronics & Safety Systems Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Fire Sensing

## 1.1 The Physics of Smoke Detection (Optical)
Most marine smoke detectors are "Photoelectric" (Optical) type.
*   **The Physics:** Inside the sensor, a light source (Infrared LED) and a receiver (Photocell) are positioned so they don't see each other. When smoke particles enter the chamber, they **Scatter the Light** onto the receiver.
*   **Response Physics:** Optical sensors are best at detecting "Smoldering Fires" (large smoke particles) like those from electrical insulation or furniture.

## 1.2 The Physics of Heat Detection
Heat detectors are used in areas where smoke is normal (e.g., Galley, Engine Room).
*   **Fixed Temperature Physics:** Trips at a specific temperature (usually 58°C or 75°C) using a bimetallic strip or a thermistor.
*   **Rate-of-Rise (RoR) Physics:** Trips if the temperature rises faster than a set rate (e.g., 8°C per minute). **Advantage:** Detects a fast-growing fire before the fixed limit is reached.

## 1.3 Flame Detection Physics (UV/IR)
Used in high-risk areas like the Purifier room or Paint locker.
*   **The Physics:** Sensors detect the specific electromagnetic radiation emitted by fire (Ultraviolet or Infrared flickering). 
*   **Response Time:** Extremely fast (milliseconds). Essential for areas where a "Flash Fire" could occur.

---

# Part 2 – Major Components & System Layout

## 2.1 The Main Control Panel (FACP)
Usually located on the Bridge.
*   **Function:** Monitors the "Loops," displays the location of any fire (e.g., "Deck 3, Cabin 14"), and controls the general alarm bells and fire doors.

## 2.2 Addressable Loops
*   **The Physics:** Modern systems use "Digital Loops." Each detector has a unique ID (Address). 
*   **Redundancy:** The loop starts and ends at the panel. **Physics:** If the cable is cut in one place, the panel can still talk to all sensors by sending data from both ends of the loop.

## 2.3 Manual Call Points (Break-Glass)
Located at all escape exits and bridge wings. **Physics:** Uses a "Normally Closed" circuit. Breaking the glass (or pushing the plastic) opens the circuit, triggering an instant "Fire" status.

## 2.4 Interface (The "VDR" and "BAMS")
The system sends every alarm and fault event to the Voyage Data Recorder (VDR). **Physics:** Crucial for post-accident investigation.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Panel Status:** "Power ON" (Green) and "System Ready" (Green). 
*   **The "Silent" Panel:** No active alarms or fault messages.
*   **Loop Load:** Loop voltage is stable (usually 24V – 30V DC).

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Loop Voltage** | 24.0 V – 28.5 V | < 18 V (Loop Fault) |
| **Detector Sensitivity**| 2.0% – 4.0% obs/m | > 10% (Dirty Sensor) |
| **Earth Leakage** | > 1.0 MΩ | < 0.1 MΩ (Earth Fault) |
| **Battery Voltage** | 26.5 V – 27.5 V | < 23 V (Battery Low) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Loop Fault" (Short or Open Circuit)
*   **Symptom:** Panel shows "Loop 1 Fault - 20 devices missing."
*   **Root Cause:** Water ingress into a detector on deck or a loose terminal in a junction box.
*   **Physics:** Vibration and salt-mist eventually corrode the tiny copper pins in the detector base.

## 4.2 Earth Fault (Ground Fault)
*   **Symptom:** "Earth Fault" lamp glows; system becomes unstable.
*   **Root Cause:** A cable rubbing against the hull or a moisture leak in a manual call point on the open deck.

## 4.3 "Detector Dirty" Warning
*   **Symptom:** Yellow warning on the panel for a specific cabin.
*   **Root Cause:** Dust, lint, or insects inside the sensing chamber.
*   **Physics:** The panel measures the "Baseline" light scatter. If it increases due to dust, the sensor is closer to its trip point, leading to a **False Alarm.**

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Smoke in a Can" Test
*   **Trade Trick:** Never test a detector with a cigarette or a real flame. **Method:** Use a certified "Aerosol Smoke Tester." **Expert Insight:** Spray the aerosol from 30cm away for only 1 second. If you spray too much, you will "coat" the sensor in oil, causing it to fail a month later.

## 5.2 Finding a "Short Circuit" via Isolators
*   **Expert Insight:** Modern loops have "Short Circuit Isolators" every 10–20 devices. **Trade Trick:** If the loop is shorted, the isolators will "pop" open. Look at the detectors; the one *between* the last working device and the first dead device is where your short circuit is located.

## 5.3 The "Hairdryer" Trick for Damp Sensors
*   **Trade Trick:** If a deck detector is giving constant "Fault" alarms after a storm, remove the head and use a hairdryer (on low heat) to dry the base and the circuit board. **Expert Insight:** 90% of deck faults are caused by condensation inside the housing.

## 5.4 Using "Inhibit" for Maintenance
*   **Expert Insight:** During "Hot Work" (welding), always **Inhibit** the specific zone on the fire panel. **Warning:** Never forget to "Enable" it at the end of the day. **Trade Trick:** Leave your cabin key on the fire panel keyboard; you can't go to sleep without noticing the panel is inhibited!

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine (Mandatory)
*   **Indicator Check:** Verify all LEDs and the buzzer.
*   **Test Call Point:** Trigger one manual call point (rotating through all zones over a year) to ensure the alarm bells ring.

## 6.2 Annual Performance Test (APT)
*   **Requirement:** 100% of all detectors must be tested once a year.
*   **Tasks:** Test every smoke, heat, and flame detector using the correct test medium. Verify the "Fire Door" release and "Fan Shutdown" functions.

## 6.3 Battery Replacement
*   **Method:** Replace the FACP backup batteries every 2 years. They must be able to power the system for **at least 24 hours** plus 30 minutes of full alarm.

---

# Part 7 – Miscellaneous Knowledge

*   **Accommodation Fans:** The fire detection system is interlocked with the accommodation AC. If fire is detected, the fans stop to prevent smoke from being spread to other cabins.
*   **General Alarm Interface:** The fire panel can trigger the ship's General Alarm (7 short and 1 long blast) automatically or manually.

**End of Document**
