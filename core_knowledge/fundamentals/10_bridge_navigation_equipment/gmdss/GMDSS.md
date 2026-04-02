# MARINE GMDSS – COMPLETE CORE KNOWLEDGE

**Equipment:** Global Maritime Distress and Safety System (e.g., JRC, Furuno RC-1800, Sailor 6000, McMurdo)

**Folder Name:** gmdss

**Prepared by:** Expert Marine Electronics & Communication Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Marine Radio

## 1.1 The Physics of Radio Frequency (RF) Propagation
GMDSS uses three primary bands to ensure communication in any part of the world:
1.  **VHF (Very High Frequency - 156 MHz):** 
    *   **Physics:** Line-of-sight. Approx. 20–30 miles. 
    *   **Use:** Short-range ship-to-ship and harbor.
2.  **MF/HF (Medium/High Frequency - 2 MHz to 30 MHz):** 
    *   **Physics:** Uses the **Ionosphere** to "bounce" signals over the horizon (Sky-wave). 
    *   **Use:** Long-range (100–3000 miles). MF is for Sea Area A2; HF is for Sea Area A3/A4.
3.  **L-Band (Satellite - 1.5 GHz to 1.6 GHz):** 
    *   **Physics:** Direct link to Geostationary (Inmarsat) or LEO (Iridium) satellites. 
    *   **Use:** Instant worldwide communication.

## 1.2 Digital Selective Calling (DSC) Physics
*   **The Physics:** DSC is a digital protocol sent via radio. 
*   **Advantage:** It allows a ship to "page" another ship or a shore station using its unique **MMSI** (Maritime Mobile Service Identity).
*   **Distress Physics:** A DSC distress alert automatically includes the ship's ID, position (from GPS), and the nature of the distress. It broadcasts continuously until acknowledged.

## 1.3 Antenna Physics (VSWR and Grounding)
*   **The Physics:** The antenna must be "Resonant" at the transmitting frequency. 
*   **VSWR (Voltage Standing Wave Ratio):** If the antenna is damaged, energy is reflected back into the radio. **Limit:** VSWR > 2.0:1 will reduce power and eventually fry the transmitter.
*   **Grounding:** HF radios require a massive copper "Ground Strap" connected to the hull. **Physics:** Without a good ground, the hull cannot act as the "other half" of the antenna, and range will be zero.

---

# Part 2 – Major Components & System Layout

## 2.1 The GMDSS Console
The integrated station on the bridge containing:
*   **VHF DSC Radio:** Usually two independent units.
*   **MF/HF DSC Radio:** With an automatic antenna tuner (ATU).
*   **Inmarsat-C (Sat-C):** A low-speed data terminal for text messages and distress alerts.

## 2.2 Emergency Position Indicating Radio Beacon (EPIRB)
*   **Physics:** Operates on **406 MHz**. When activated, it sends a signal to the COSPAS-SARSAT satellites.
*   **Locating:** Includes a **121.5 MHz** "Homing" signal for rescue aircraft and a built-in GPS.

## 2.3 Search and Rescue Transponder (SART)
*   **Physics:** Reacts to X-band Radar (9 GHz). 
*   **Display:** When a ship's radar hits the SART, it shows **12 dots** in a line on the radar screen, pointing directly to the survivor's location.

## 2.4 NAVTEX (Navigational Telex)
*   **Physics:** Operates on **518 kHz**. An automatic receiver that prints safety information (Weather, Nav-warnings).

## 2.5 Battery Reserve (GMDSS Batteries)
A mandatory dedicated battery bank. **Physics:** Must power the GMDSS console for **1 hour** (if the emergency generator works) or **6 hours** (if no emergency generator is fitted).

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **NAVTEX:** Printing clear messages without "corrupted" characters.
*   **Sat-C:** Status shows "Logged In" to a specific Ocean Region satellite.
*   **DSC:** "Watch Receiver" is active on Channel 70 (VHF) and 2187.5 kHz (MF).

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **VHF TX Power** | 25 Watts | < 1 Watt (Low Power) |
| **Antenna VSWR** | < 1.5 : 1 | > 3.0 : 1 (Antenna Fail) |
| **Battery Voltage** | 24V – 27V | < 22V (Battery Fail) |
| **GPS Input** | Valid Pos/Time | Missing (Manual Input)|

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Position Lost" on GMDSS
*   **Symptom:** Sat-C and VHF show "No GPS" or a position from 3 days ago.
*   **Root Cause:** Failed serial data interface from the Bridge GPS.
*   **Danger:** If you press the Distress button, the rescue authorities won't know where you are.

## 4.2 HF ATU "Tuning Fail"
*   **Symptom:** MF/HF radio shows "Tune Error" when trying to transmit.
*   **Root Cause:** Corrosion on the HF whip antenna base or salt-buildup on the insulators.
*   **Physics:** The ATU cannot find a resonance point because the antenna's electrical characteristics have changed due to salt-leakage to the hull.

## 4.3 EPIRB "False Alarm"
*   **Symptom:** Coast Guard calls the ship reporting a distress signal.
*   **Root Cause:** Water ingress into the EPIRB housing or testing the unit outside its protective bracket.
*   **Expert Insight:** 95% of EPIRB alerts are false. **Immediate Action:** Contact the nearest Coast Radio Station and cancel the alert.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Weekly Test" (The DSC Loopback)
*   **Trade Trick:** Once a week, perform a "DSC Internal Test." **Expert Insight:** You can also perform a "Safety Test Call" to a nearby shore station. **Do not use the Distress button!** Use the "Individual Call" function to verify the transmitter and receiver.

## 5.2 Cleaning "High-Voltage" Insulators
*   **Trade Trick:** If the HF radio is weak in rainy weather, wash the large white insulators at the base of the whip antenna with **Fresh Water and a little soap**. **Physics:** Salt-crust becomes conductive when wet, "shorting" your radio energy directly to the ship's hull.

## 5.3 SART "Radar Test"
*   **Trade Trick:** To test a SART without triggering a rescue, take it to the bridge and put it in "Test Mode." Turn on the ship's X-band Radar. **Expert Insight:** You should see **concentric circles** on the radar screen. If you only see dots, the SART is weak.

## 5.4 Checking Battery "Internal Resistance"
*   **Expert Insight:** GMDSS batteries sit on "Float Charge" for years. **Trade Trick:** Measure the voltage while the VHF is transmitting at full power (25W). **If the voltage drops by more than 1.0V**, the battery is "Soft" and will fail within minutes during a real blackout.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine (Mandatory)
*   **DSC Test:** Verify internal self-tests of VHF and MF/HF.
*   **NAVTEX:** Check for paper/ink or digital log status.
*   **Reserve Power:** Test the batteries under load (without the charger).

## 6.2 Annual Radio Survey (Mandatory)
*   **Requirement:** A certified radio surveyor must test every frequency and power level. They will check the **EPIRB Hydrostatic Release** and the **SART battery** dates.
*   **Shore-Based Maintenance (SBM):** The ship must have a valid SBM certificate from a service company.

## 6.3 Battery Replacement
*   **Method:** Replace GMDSS batteries every 2-4 years depending on the type. **Safety:** Never mix old and new batteries in the same bank.

---

# Part 7 – Miscellaneous Knowledge

*   **Sea Areas:** 
    *   **A1:** Within VHF range of shore (~20 miles).
    *   **A2:** Within MF range of shore (~100 miles).
    *   **A3:** Within Satellite range (Worldwide except Poles).
    *   **A4:** The Polar regions (requires HF or Iridium).
*   **MMSI:** A 9-digit number. The first 3 digits are the "MID" (Maritime Identification Digits) which tell you the ship's flag state (e.g., 232 for UK, 366 for USA).

**End of Document**
