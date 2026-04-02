# MARINE GPS – COMPLETE CORE KNOWLEDGE

**Equipment:** Global Positioning System / GNSS Receiver (e.g., JRC JLR-21, Furuno GP-170, Simrad GN70, Saab R5)

**Folder Name:** gps

**Prepared by:** Expert Marine Electronics & Satellite Systems Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of GNSS

## 1.1 The Physics of Trilateration
GPS (and the broader GNSS - Global Navigation Satellite Systems) determines position by measuring the distance to at least four satellites.
*   **The Physics:** Range $R = c \cdot (t_{rcv} - t_{sat})$. 
    Where $c$ is the speed of light and $t$ is the timestamp. By intersecting four "Spheres" of distance, the receiver calculates Latitude, Longitude, Altitude, and Time.
*   **Atomic Clock Physics:** Satellites use atomic clocks (accurate to nanoseconds). The receiver uses a cheaper quartz clock, which is why the **4th Satellite** is required to solve for the receiver's "Clock Bias."

## 1.2 Signal Propagation Physics (Errors)
*   **Ionospheric Delay:** As the radio signal (L1 band - 1575.42 MHz) passes through the Ionosphere, it slows down due to free electrons. **Physics:** This is the largest source of error (up to 5 meters).
*   **Multipath Effect:** In port, the signal can bounce off cranes or containers before reaching the antenna. **Physics:** This increases the "Time of Flight," causing the position to jump.

## 1.3 DGPS and SBAS Physics
*   **DGPS (Differential GPS):** A shore station at a known location calculates the error in the satellite signals and broadcasts a correction.
*   **SBAS (Satellite Based Augmentation System):** (e.g., WAAS, EGNOS). Corrections are sent via geostationary satellites. **Physics:** Improves accuracy from 10 meters down to **less than 1 meter**.

---

# Part 2 – Major Components & System Layout

## 2.1 The GNSS Antenna
Usually a "Patch Antenna" or a "Helix Antenna" mounted on the monkey island.
*   **Physics:** It must have a clear 360° view of the horizon. It contains a "Low Noise Amplifier" (LNA) to boost the incredibly weak satellite signals ($10^{-16}$ Watts).

## 2.2 The Receiver Unit (The Processor)
Performs the complex "Correlator" math to track multiple satellite codes (C/A codes) simultaneously.

## 2.3 The Display and Interface Unit
Shows the position, COG (Course Over Ground), SOG (Speed Over Ground), and the **DOP (Dilution of Precision)**.

## 2.4 Time Synchronization (The Master Clock)
GPS provides the "UTC Time" for the entire ship. **Physics:** All bridge systems (Radar, VDR, ECDIS) synchronize their internal clocks to the GPS pulse-per-second (PPS) signal.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Fix Status:** "3D Fix" or "Differential Fix" is displayed.
*   **Satellites:** At least 8–12 satellites are being tracked with good signal strength.
*   **DOP:** The HDOP (Horizontal Dilution of Precision) is **< 1.5**. (Lower is better).

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **HDOP** | 0.8 – 1.5 | > 4.0 (Poor Geometry) |
| **Signal-to-Noise (SNR)**| 35 – 50 dBHz | < 25 dBHz (Weak Signal) |
| **Position Accuracy** | < 5 meters | > 100 meters (No Fix) |
| **Correction Source** | SBAS / Beacon | "Internal" (Uncorrected) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Position Lost" Alarm
*   **Symptom:** Display shows "No Fix" or Latitude/Longitude is blank.
*   **Root Cause:** Water ingress into the antenna or cable (Corrosion).
*   **Physics:** Saltwater in the coaxial cable increases the "Attenuation" (Signal Loss) so much that the receiver cannot see the satellites.

## 4.2 SBAS / Differential Not Available
*   **Symptom:** Accuracy drops; "DGPS Lost" warning.
*   **Root Cause:** Ship is in an area not covered by SBAS (e.g., deep South Atlantic) or the DGPS beacon receiver is out of range.

## 4.3 Signal Interference (Jamming/Spoofing)
*   **Symptom:** Position jumps wildly or shows the ship is in the middle of a desert.
*   **Root Cause:** Proximity to naval exercises or "Electronic Warfare" zones.
*   **Physics:** A stronger "False" signal overrides the weak satellite signals.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Antenna Voltage" Test
*   **Trade Trick:** If the GPS signal is weak, disconnect the cable from the receiver. Use a multimeter to measure the DC voltage coming *out* of the receiver's antenna port. **It should be 5V or 12V DC.** If the voltage is there but the signal is gone, the antenna's internal amplifier is "Fried" (often by a nearby lightning strike).

## 5.2 Detecting "Multipath" in Port
*   **Expert Insight:** If the GPS speed shows "0.5 knots" while the ship is tied up, you have **Multipath Interference**. **Trade Trick:** Look at the satellite map on the GPS display. If all the "Green" satellites are on one side of the sky, the ship's cranes are blocking the other half.

## 5.3 The "Cold Start" Reset
*   **Trade Trick:** If the GPS hasn't been used for months and won't lock, perform a **"Cold Start" or "Factory Reset."** This forces the receiver to download a new **Almanac** (The satellite map). **Warning:** It can take up to 20 minutes to get the first fix after a cold start.

## 5.4 Using "GNSS" (Glonass/Galileo/Beidou)
*   **Expert Insight:** Most modern "GPS" units are actually **Multi-constellation receivers**. **Trade Trick:** If you are in a narrow fjord or between high mountains, enable **GLONASS** (Russian) or **Galileo** (EU) in the settings. This doubles the number of available satellites, providing a fix where standard GPS would fail.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Cross-Check:** Verify GPS position against a Radar fix or a visual bearing.
*   **Battery:** Check the memory backup battery life (if the GPS loses its "Home" position every time it's turned off, the battery is dead).

## 6.2 Annual Performance Test (APT)
*   **Requirement:** Verify the NMEA 0183 output to the VDR and ECDIS. Ensure the "Datum" is locked to **WGS-84**.

## 6.3 Antenna Physical Check
*   **Check:** Look for cracks in the white plastic dome (Radome). **Expert Insight:** If the dome is cracked, moisture will enter and slowly destroy the circuit board. Seal any tiny cracks with marine silicone immediately.

---

# Part 7 – Miscellaneous Knowledge

*   **RAIM (Receiver Autonomous Integrity Monitoring):** A software check that ensures the satellites aren't "Lying." It requires 5 satellites to detect a bad signal and 6 to "exclude" it.
*   **Leap Seconds:** GPS time is continuous, but UTC time has "Leap Seconds." The GPS receiver must automatically apply the current offset (e.g., 18 seconds) to provide the correct bridge time.

**End of Document**
