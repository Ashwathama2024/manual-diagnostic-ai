# MARINE ECDIS – COMPLETE CORE KNOWLEDGE

**Equipment:** Electronic Chart Display and Information System (e.g., Transas/Wärtsilä Navi-Sailor, JRC, Furuno FMD, Sperry VisionMaster)

**Folder Name:** ecdis

**Prepared by:** Expert Marine Electronics & Navigation Systems Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Digital Charting

## 1.1 The Physics of Geodetic Datums (WGS-84)
ECDIS plots the ship's position on a digital map. 
*   **The Physics:** The Earth is not a perfect sphere; it is an oblate spheroid. ECDIS uses the **WGS-84 (World Geodetic System 1984)** datum as its mathematical model of the Earth.
*   **Datum Shift Physics:** If a GPS position (WGS-84) is plotted on an old chart using a different datum (e.g., Tokyo Datum), the ship's position could be hundreds of meters off. **ECDIS eliminates this risk by ensuring all ENC charts use WGS-84.**

## 1.2 Vector vs. Raster Physics
1.  **ENC (Vector Charts):** 
    *   **Physics:** Charts are built from a database of objects (points, lines, areas). 
    *   **Advantage:** "Intelligent" charting. The computer knows that a specific blue area is "Shallow Water" and can trigger an alarm if the ship's safety contour enters it.
2.  **RNC (Raster Charts):** 
    *   **Physics:** A simple digital "photocopy" of a paper chart. The computer doesn't "know" what the pixels represent. Used only as a backup when ENCs are unavailable.

## 1.3 Safety Contour Physics
The ECDIS uses the ship's **Dynamic Draft** to calculate safe areas.
*   **The Physics:** Safety Contour = Static Draft + Squat + Safety Margin + Cat. allowance.
*   **Visual Display:** ECDIS colors the water differently based on this contour (Dark Blue for unsafe, Light Blue for intermediate, White for safe).

---

# Part 2 – Major Components & System Layout

## 2.1 The Processor and Graphics Engine
Industrial PC (IPC) with dual power supplies. It must handle high-speed graphics rendering without "Lag."

## 2.2 Sensor Integration (The Interface)
ECDIS is the "Central Hub" for bridge data via NMEA 0183 or IEC 61162 Ethernet:
*   **GPS:** Position and Time.
*   **Gyro:** True Heading.
*   **Log:** Speed through water (STW).
*   **AIS:** Target data.
*   **Radar:** Echo overlay.

## 2.3 Uninterruptible Power Supply (UPS)
A mandatory battery backup that keeps the ECDIS alive for at least 45 seconds (to bridge the gap until the emergency generator starts) and up to 30 minutes for a safe shutdown.

## 2.4 Redundancy (The Dual ECDIS)
To be "Paperless," a ship must carry two independent ECDIS units (Master and Backup) with independent power and sensor feeds.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Update Status:** All charts are updated to the latest "Notice to Mariners" (Weekly).
*   **Position Check:** The GPS position matches the Radar echo of a known landmark (Coastline).
*   **Alarms:** No "Red" alarms. "Yellow" warnings (e.g., missing sensors) are acknowledged.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Safety Contour** | Draft + ~2.0m | Automatic Alarm |
| **Look-ahead Time** | 6 – 15 minutes | Zero (Alarm Off) |
| **Position Source** | GPS 1 (Differential) | GPS Failure (Dead Reckoning) |
| **Chart Scale** | 1:1 (Original Scale) | Over-scale / Under-scale |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Position Jump" (GNSS Error)
*   **Symptom:** The ship icon suddenly jumps 500 meters or "zig-zags" on the screen.
*   **Root Cause:** GPS multipath interference (in port) or "Spoofing" in sensitive areas.
*   **Danger:** Auto-pilot will follow the "False" position, potentially driving the ship aground.

## 4.2 Chart Update Failure
*   **Symptom:** The ECDIS won't accept new ENC cells from the USB or Internet.
*   **Root Cause:** Corrupted "Permit" file or an expired S-63 security certificate.

## 4.3 Alarm Fatigue (Nuisance Alarms)
*   **Symptom:** Constant "Crossing Safety Contour" alarms in a narrow channel.
*   **Root Cause:** Safety depth set too deep for the port environment or the "Look-ahead" sector is too wide.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Cross-Check" Routine
*   **Trade Trick:** Never trust the GPS alone. Every hour, perform a **Radar Overlay** check. If the radar land-echo doesn't line up with the chart coastline, your GPS has a "Static Offset." **Expert Insight:** Use the manual "Offset" function to align them.

## 5.2 Detecting "Over-scale" Errors
*   **Trade Trick:** If you zoom in too much, the chart becomes blurry or warns "Over-scale." **Expert Insight:** Important details (like a tiny rock) might be hidden if you aren't using the chart at its **Compilation Scale**. Always press the "Standard Scale" button when in doubt.

## 5.3 Manual "Dead Reckoning" (DR)
*   **Expert Insight:** If the GPS fails, the ECDIS should automatically switch to DR mode using the Gyro and Log. **Trade Trick:** You must manually enter the **Estimated Current and Leeway** every 30 minutes to maintain accuracy.

## 5.4 The "Hard Reboot" Caution
*   **Trade Trick:** If the ECDIS software freezes, don't just pull the power. Try the keyboard shortcut (e.g., Ctrl+Alt+Shift+Q for some units) to force a software reset. **Expert Insight:** Pulling the power can corrupt the "Log Files" required by investigators in case of an accident.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Maintenance
*   **ENC Updates:** Download and apply the latest ENC updates via the "Service Provider" (e.g., Admiralty, NAVTOR).
*   **Log Backup:** Ensure the "Voyage Log" is being backed up to an external drive.

## 6.2 Annual Performance Test (APT)
*   **Requirement:** An authorized service technician must perform an APT. They will check the NMEA telegram integrity and the battery health of the UPS.

## 6.3 Cleaning the Cooling Fans
*   **Method:** ECDIS units are often in enclosed bridge consoles. **Expert Insight:** Vacuum the air intake filters every 3 months. Overheating is the #1 cause of "Screen Freezing" during critical maneuvers.

---

# Part 7 – Miscellaneous Knowledge

*   **SCAMIN (Scale Minimum):** A feature that hides small objects as you zoom out to reduce screen clutter. **Danger:** If not understood, an officer might zoom out and "lose" a dangerous rock.
*   **S-57 and S-100:** S-57 is the current data standard. S-100 is the new "Next Generation" standard that allows for real-time tide and weather overlays.

**End of Document**
