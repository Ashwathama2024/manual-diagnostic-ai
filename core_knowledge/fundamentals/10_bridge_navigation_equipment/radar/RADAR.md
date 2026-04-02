# MARINE RADAR – COMPLETE CORE KNOWLEDGE

**Equipment:** Marine Navigation Radar and ARPA System (e.g., Furuno, JRC, Sperry Marine, Raytheon Anschütz)

**Folder Name:** radar

**Prepared by:** Expert Marine Electronics & Navigation Systems Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Radar

## 1.1 The Physics of Pulse-Echo
Radar works by emitting a high-power microwave pulse and measuring the time it takes for the echo to return from a target.
*   **The Physics:** Range $D = \frac{c \cdot t}{2}$. 
    Where $c$ is the speed of light ($300,000$ km/s) and $t$ is the elapsed time. Because $c$ is so large, the electronics must measure nanoseconds ($10^{-9}$ s) to achieve meter-level accuracy.
*   **Pulse Width Physics:** 
    *   **Short Pulse:** Used for high-resolution at close range (e.g., in a harbor).
    *   **Long Pulse:** Used for maximum energy to detect targets at long distances (e.g., 48+ miles).

## 1.2 The Physics of Wavelength (X-band vs. S-band)
1.  **X-band (3cm / 9.4 GHz):** 
    *   **Physics:** Shorter wavelength provides higher resolution and better detection of small targets (buoys, ice). 
    *   **Weakness:** Easily scattered by rain and sea spray (Clutter).
2.  **S-band (10cm / 3.0 GHz):** 
    *   **Physics:** Longer wavelength "cuts through" rain and fog much better. 
    *   **Strength:** Essential for long-range early warning in heavy weather.

## 1.3 Beamwidth and Resolution Physics
*   **Horizontal Beamwidth:** Determined by the length of the antenna. A longer scanner (e.g., 8ft vs 4ft) creates a narrower beam.
*   **The Physics:** A narrower beam provides better **Bearing Resolution**, allowing the radar to distinguish between two ships that are close together at the same range.

---

# Part 2 – Major Components & System Layout

## 2.1 The Magnetron (The Transmitter)
*   **Physics:** A vacuum tube that converts DC electrical energy into high-power microwave pulses. It is a "Consumable" component with a life of approx. 3,000–8,000 hours.

## 2.2 The Waveguide and Scanner
*   **Waveguide:** A hollow copper tube that "conducts" the microwave energy from the transmitter to the antenna with minimal loss.
*   **Scanner (Antenna):** Rotates at 24–48 RPM to provide a 360° view.

## 2.3 The Receiver and Signal Processor
*   **Duplexer:** A high-speed switch that protects the delicate receiver while the powerful transmitter is firing.
*   **Digital Processing:** Converts the analog echoes into digital "cells" for display and ARPA tracking.

## 2.4 ARPA (Automatic Radar Plotting Aid)
*   **Physics:** The computer tracks the change in a target's position over multiple rotations to calculate its **Course (COG)**, **Speed (SOG)**, **CPA (Closest Point of Approach)**, and **TCPA (Time to CPA)**.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Tuning:** The "Tuning Bar" on the display is at maximum when aimed at a distant target.
*   **Image Clarity:** Coastlines match the chart; targets are sharp and don't "smear" unless the ship is turning.
*   **Performance Monitor:** When activated, shows a clear set of "concentric rings" or a specific pattern, proving the magnetron power and receiver sensitivity are within spec.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Magnetron Current** | Stable (e.g., 4-6mA) | Fluctuating (Aging) |
| **Scanner RPM** | 24 or 42 RPM | Low RPM (Motor Fail) |
| **Heading Flash** | Aligned with Bow | Misaligned (Bearing Error)|
| **CPA Alarm Limit** | 0.5 – 2.0 miles | Target within CPA |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Weak" Targets (Low Sensitivity)
*   **Symptom:** Distant land or large ships only appear at 6 miles instead of 24 miles.
*   **Root Cause:** Aging magnetron or water ingress into the waveguide.
*   **Physics:** The output power has dropped, or the returning echo is being absorbed by moisture before it reaches the receiver.

## 4.2 Heading Marker Error
*   **Symptom:** Targets appear 5° to the left or right of their actual position.
*   **Root Cause:** The "Heading Flash" sensor in the scanner unit has moved or the scanner belt is slipping.
*   **Danger:** Massive risk of collision if the officer relies on the radar bearing for "Rules of the Road" decisions.

## 4.3 Sea and Rain Clutter (Blindness)
*   **Symptom:** The center of the screen is a "White Cloud" of noise.
*   **Root Cause:** Improper use of the "Anti-Clutter" controls.
*   **Physics:** High waves reflect radar energy. Over-adjusting the clutter control will "suppress" real small targets (like a wooden fishing boat) along with the waves.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Performance Monitor" (PM) Test
*   **Trade Trick:** Never take over a watch without checking the PM. **If the PM arc is shorter than the value recorded in the manual**, your radar is effectively blind to small targets. **Expert Insight:** A failing magnetron often works fine at close range but fails the PM test.

## 5.2 The "Ghost Target" Detection
*   **Trade Trick:** If you see a target that follows your ship's movement exactly, it is a "Ghost" (Multiple Reflection). **Expert Insight:** Check if the target is on the same bearing as your own ship's funnel or a large crane. The radar pulse is bouncing off your own ship's structure.

## 5.3 Manual Tuning vs. Auto-Tuning
*   **Trade Trick:** If the radar is not detecting distant targets well in "Auto," switch to **Manual Tuning**. Turn the knob until the "Background Noise" (speckles) is just visible. This is the most sensitive the receiver can be.

## 5.4 Detecting Waveguide Leaks
*   **Expert Insight:** If the radar works in dry weather but fails in rain, you have a leak in the waveguide joints. **Trade Trick:** Look for "Green Corrosion" (Verdigris) at the waveguide flanges. Any moisture here will reflect the energy back into the transmitter, eventually killing the magnetron.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Visual:** Check the scanner for any ropes or birds' nests.
*   **Display:** Clean the monitor with a micro-fiber cloth. **Never use window cleaner**, as it strips the anti-glare coating.

## 6.2 Annual Magnetron Replacement
*   **Method:** Record the "Magnetron Run Hours." Most companies replace the magnetron proactively every 5,000 hours. **Safety:** After replacement, you MUST re-align the Heading Marker and perform a new PM baseline.

## 6.3 Scanner Motor and Belt
*   **Check:** Every 6 months, open the scanner pedestal. Check the drive belt tension and grease the motor gears. A snapped belt in a storm means you are navigating blind.

---

# Part 7 – Miscellaneous Knowledge

*   **RACON (Radar Beacon):** A buoy or lighthouse that "talks back" to your radar. It shows up as a "Morse Code" dash on your screen (e.g., a "K" for a specific buoy).
*   **AIS Overlay:** Modern radars overlay AIS data (triangles) on the radar targets. **Warning:** Always trust the radar "Echo" over the AIS triangle. The AIS relies on the other ship's GPS, which could be wrong. The Radar echo is your own physical measurement.

**End of Document**
