# MARINE AIS – COMPLETE CORE KNOWLEDGE

**Equipment:** Automatic Identification System (e.g., JRC, Furuno FA-170, Saab R5, McMurdo)

**Folder Name:** ais

**Prepared by:** Expert Marine Electronics & Communication Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of AIS

## 1.1 The Physics of VHF Propagation
AIS operates in the VHF maritime mobile band (161.975 MHz and 162.025 MHz).
*   **The Physics:** Propagation is primarily **Line-of-Sight**. 
*   **Range:** Typically 20–30 nautical miles depending on antenna height. 
*   **Atmospheric Ducting:** In certain weather conditions (Temperature Inversions), AIS signals can "bounce" and be received from hundreds of miles away.

## 1.2 Slot Reservation Physics (SOTDMA)
With thousands of ships in a small area (like the English Channel), how do they all talk without jamming each other?
*   **The Physics:** SOTDMA (Self-Organizing Time Division Multiple Access). 
*   **The Process:** Each minute is divided into **2,250 time slots**. A Class A AIS unit "listens" to the traffic and "reserves" an empty slot for its own transmission. It then tells other ships which slot it will use next.

## 1.3 Data Types and Update Rates
AIS transmits three types of data:
1.  **Static Data:** MMSI, IMO Number, Ship Name, Dimensions (Fixed).
2.  **Dynamic Data:** GPS Position, COG, SOG, Heading (Variable).
3.  **Voyage Data:** Destination, ETA, Draft, Cargo Type (Manual input).
*   **Update Physics:** Dynamic data updates every **2 to 10 seconds** depending on the ship's speed and rate of turn.

---

# Part 2 – Major Components & System Layout

## 2.1 The AIS Transponder
A specialized radio that contains two VHF receivers, one VHF transmitter, and an internal GPS receiver (used primarily for timing).

## 2.2 VHF and GPS Antennas
*   **VHF Antenna:** Usually a vertical whip antenna mounted as high as possible.
*   **GPS Antenna:** A dedicated passive or active antenna with a clear view of the sky. **Physics:** The AIS requires UTC time from GPS to synchronize its SOTDMA time slots.

## 2.3 The Pilot Plug
A standardized 9-pin connector on the bridge.
*   **Function:** Allows Marine Pilots to plug in their own laptop/tablet to receive real-time AIS and GPS data for the ship.

## 2.4 Interface (The "Long-Range" Link)
AIS is connected to the ship's **Sat-C** or **LRIT** system. **Physics:** Allows the AIS data to be sent via satellite to shore authorities when the ship is far out at sea (beyond VHF range).

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **LED Status:** "Power" is Green, "TX" flashes amber every few seconds, "Error" is OFF.
*   **Target List:** The display shows a list of nearby ships with their names and ranges.
*   **Position:** Internal GPS shows a valid latitude/longitude.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **VSWR (Antenna)** | < 1.5 : 1 | > 3.0 : 1 (Antenna Fault) |
| **Transmitter Power** | 12.5 Watts | < 10 Watts (Weak Signal) |
| **GPS Fix Status** | 3D Fix | No Fix (Sync Error) |
| **Heading Source** | External Gyro | Missing (DR Mode) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "TX Malfunction" Alarm
*   **Symptom:** AIS stops transmitting; other ships cannot see you.
*   **Root Cause:** Failed VHF antenna cable or a "Short Circuit" in the antenna due to salt-water ingress.
*   **Physics:** High VSWR (Voltage Standing Wave Ratio) causes the radio energy to "bounce back" into the transmitter, triggering a safety shutdown.

## 4.2 Heading/Rate-of-Turn Error
*   **Symptom:** AIS shows "No Heading" or a static heading while the ship is turning.
*   **Root Cause:** Failed NMEA interface from the Gyrocompass.
*   **Danger:** Other ships' ARPA systems will receive incorrect "Rate of Turn" data, making collision avoidance difficult.

## 4.3 Incorrect Vessel Dimensions
*   **Symptom:** Other ships report that your ship icon is 100 meters ahead of or behind your actual position.
*   **Root Cause:** Incorrect "Antenna Offset" values entered in the static data setup. 
*   **Physics:** The GPS position is at the antenna. You must tell the AIS the distance from the antenna to the Bow, Stern, Port, and Starboard sides.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "AIS-Shore" Loopback
*   **Trade Trick:** If you suspect your transmitter is weak, look at the "Station List." **Expert Insight:** Check the SNR (Signal to Noise Ratio) of the messages received from a known shore station (AIS Base Station). If your SNR is < 10dB while other ships are at 30dB, your cable is corroded.

## 5.2 Detecting "GPS Spoofing"
*   **Trade Trick:** If the AIS position jumps while the ship is at anchor, check the "Internal GPS" vs. the "Bridge GPS." **Expert Insight:** AIS units have their own internal GPS. If they disagree, you are being spoofed or have a multipath error.

## 5.3 Resetting the "Slot Collision"
*   **Trade Trick:** In extremely crowded ports, the AIS might show a "Slot Full" alarm. **Solution:** Do nothing. The SOTDMA protocol will automatically find a slot in the next minute. If it persists, **reboot the AIS** to clear the slot-reservation memory.

## 5.4 Using the "Silent Mode" (Blue Mode)
*   **Expert Insight:** Some ships have a "Silent Mode" switch. **Safety Warning:** This stops the AIS from transmitting (Receive only). **Use only in Piracy zones.** Navigating in a high-traffic area in silent mode is a major safety violation.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Dimension Check:** Verify that the "Voyage Data" (Draft, Destination, ETA) is updated. This is the #1 item checked by Port State Control.
*   **Antenna Check:** Inspect the VHF antenna for physical damage or loose mounting.

## 6.2 Annual Radio Survey (Mandatory)
*   **Requirement:** A certified Class radio surveyor must test the AIS frequency, power, and SOTDMA integrity. They will issue a **Safety Radio Certificate**.

## 6.3 Cleaning the Pilot Plug
*   **Method:** The Pilot Plug is often exposed to bridge-wing environment. **Trade Trick:** Use a cotton swab with contact cleaner to clean the pins monthly. A "noisy" pilot plug connection makes the pilot very unhappy.

---

# Part 7 – Miscellaneous Knowledge

*   **AIS-SART:** A search and rescue transmitter that shows up on AIS screens as a "Circle with a Cross." It indicates a lifeboat or person in the water.
*   **Virtual AtoN:** A navigation buoy that doesn't exist physically. It is "broadcast" by a shore station and appears on the ECDIS/Radar screen as a "Virtual Buoy."

**End of Document**
