# MARINE VDR (VOYAGE DATA RECORDER) – COMPLETE CORE KNOWLEDGE

**Equipment:** Shipboard Voyage Data Recorder (e.g., JRC JCY-1900, Furuno VR-7000, Danelec DM100, NetWave NW6000)

**Folder Name:** vdr

**Prepared by:** Expert Marine Electronics & Regulatory Compliance Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Data Recording

## 1.1 The Physics of the "Black Box"
The VDR is designed to survive a catastrophic accident (Sinking, Fire, Explosion) and preserve the last 48 hours of ship data for investigators.
*   **The Physics of Survival:** The Final Recording Medium (FRM) is housed in a "Fixed Capsule" or a "Float-free Capsule." 
*   **Thermal Protection:** The fixed capsule must withstand $1100^\circ C$ for 1 hour and $260^\circ C$ for 10 hours.
*   **Pressure Protection:** The fixed capsule must resist the pressure at **6,000 meters depth** (approx. 600 bar).

## 1.2 Data Capture Physics
The VDR records a massive amount of "Heterogeneous" data from across the ship:
1.  **Audio:** Bridge microphones and VHF radio (Physics: Multi-channel spatial recording).
2.  **Video (Radar/ECDIS):** Screen captures via DVI or Ethernet (Physics: High-resolution frame-grabbing).
3.  **Serial Data (NMEA):** GPS, Gyro, Log, AIS, AMS (Physics: Time-stamped ASCII telegrams).

## 1.3 Float-Free Physics (HRU)
*   **The Component:** Hydrostatic Release Unit (HRU).
*   **The Physics:** When the ship sinks to a depth of **2–4 meters**, water pressure triggers a knife or spring mechanism that cuts the capsule's securing strap. The capsule then floats to the surface and activates its internal **Acoustic Beacon** and **Satellite EPIRB**.

---

# Part 2 – Major Components & System Layout

## 2.1 The Data Management Unit (DMU)
The "Computer" of the system. Usually located in the bridge console.
*   **Function:** It collects data from all interfaces, compresses it, and sends it to the recording capsules.

## 2.2 The Fixed Capsule (Protective Capsule)
Bolted to the deck above the bridge. Contains a high-durability Solid State Drive (SSD).

## 2.3 The Float-Free Capsule
Mounted on the bridge wing. Identical data to the fixed capsule, but designed to float.

## 2.4 Bridge Audio System
*   **Bridge Microphones:** Strategically placed to record all conversations, commands, and alarms. **Physics:** Must record at a frequency range of 150 Hz to 3.5 kHz.

## 2.5 Battery Backup (UPS)
A mandatory requirement. **Physics:** The VDR must continue recording audio for **at least 2 hours** after the ship's emergency power fails.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **System Status:** "Recording" indicator is Green. No error messages on the display.
*   **Data Flow:** All sensor inputs (GPS, Gyro, etc.) show "Active" status in the diagnostic menu.
*   **Capsule Status:** Communication with both Fixed and Float-free capsules is confirmed.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Recording Time** | Min 48 hours | < 48 hours (SSD Fail) |
| **Microphone Check** | Clear Audio | "Hum" or Silence (Fail) |
| **Battery Voltage** | 24V – 27V | < 22V (UPS Fault) |
| **Capsule Temp** | Ambient | > 60°C (Internal Heat) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Recording Error" (Capsule Disconnect)
*   **Symptom:** VDR alarm sounds; display shows "Fixed Capsule Error."
*   **Root Cause:** Water ingress into the capsule connector or a failed Ethernet switch.
*   **Physics:** Exposed deck cables are subject to salt-spray and UV degradation. A tiny crack in the cable jacket will eventually lead to signal loss.

## 4.2 Microphone "Clipping" / Noise
*   **Symptom:** Audio playback is distorted or full of static.
*   **Root Cause:** Microphone is located too close to an AC vent or a loud bridge buzzer.

## 4.3 Sensor Timeout
*   **Symptom:** "NMEA Error" for GPS or Gyro.
*   **Root Cause:** Data buffer overflow or a failed serial-to-ethernet converter.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Save" Button (The most important task)
*   **Critical Action:** In the event of an accident (even a minor "Touch-and-Go"), the officer **MUST press the "SAVE" button** on the VDR. 
*   **Physics:** VDRs record in a "Loop." If you don't press SAVE, the crucial accident data will be **overwritten** after 48 hours.

## 5.2 The "Audio Self-Test"
*   **Trade Trick:** Every week, walk around the bridge and **clap your hands** near each microphone while the VDR is in playback/monitor mode. **Expert Insight:** Ensure you can hear the general alarm and the engine room telegraph buzzer clearly in the recording.

## 5.3 Detecting a "Weak" Acoustic Beacon
*   **Trade Trick:** The fixed capsule has an **Underwater Locating Beacon (ULB)**. **Expert Insight:** Check the battery expiry date. **Trade Trick:** Dip the capsule's test probes in a glass of saltwater to trigger the beacon and listen for the ultrasonic pulse using a portable receiver.

## 5.4 Using the "Web Interface"
*   **Trade Trick:** Most modern VDRs (like Danelec or Furuno) have a hidden web-server. **Expert Insight:** Plug a laptop into the service port. You can see real-time Radar and ECDIS screen captures to verify the recording quality without using the main VDR screen.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Indicator Check:** Verify the Green recording LED.
*   **HRU Check:** Check the expiry date on the float-free hydrostatic release.

## 6.2 Annual Performance Test (APT) - Mandatory
*   **Requirement:** An authorized service company must perform the APT within 3 months of the ship's certificate anniversary.
*   **Tasks:** Full data download, audio quality check, and capsule integrity verification. **Result:** The "VDR Performance Test Certificate" is a mandatory ship's document.

## 6.3 Beacon Battery Replacement
*   **Method:** Replace the ULB battery every 3 years. **Safety:** The beacon must be able to pulse for **90 days** under water.

---

# Part 7 – Miscellaneous Knowledge

*   **S-VDR:** Simplified VDR used on older cargo ships. It records less data (e.g., no AIS or hull stress data).
*   **Legal Ownership:** The data on the VDR belongs to the ship owner, but in an accident, the **Flag State** or **Port State** investigators have the legal right to seize the capsules.

**End of Document**
