# MARINE ELECTRICAL LIGHTING SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Shipboard Main, Emergency, and Navigation Lighting Systems (e.g., Glamox, Aqua Signal, Peters & Bey, Philips)

**Folder Name:** lighting_system

**Prepared by:** Senior Marine Electrical Engineer & Automation Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Marine Illumination

## 1.1 The Physics of Luminous Efficacy (LED vs. Old Tech)
Modern ships are rapidly converting to LED (Light Emitting Diode) technology.
*   **The Physics:** LEDs convert electricity directly into light with minimal heat. **Comparison:** 
    *   **Incandescent:** 15 lumens/watt (most energy lost as heat).
    *   **Fluorescent:** 60–80 lumens/watt.
    *   **LED:** 120–150 lumens/watt.
*   **Maintenance Physics:** LEDs have a lifespan of 50,000+ hours and are immune to the high-frequency vibrations found in engine rooms that kill incandescent filaments.

## 1.2 The Physics of Emergency Lighting (Redundancy)
Lighting is a critical safety system divided into three physical circuits:
1.  **Main Lighting (440V/220V AC):** Powered by main generators.
2.  **Emergency Lighting (220V AC):** Powered by the Emergency Switchboard. **Physics:** Must switch on automatically within 45 seconds of a blackout.
3.  **Transitional / Battery Lighting (24V DC):** **Physics:** Powered by independent battery banks. Must be instantaneous, providing enough light for the crew to reach the escape routes during the "Blackout Gap."

## 1.3 Navigation Lights Physics (COLREGs)
*   **The Physics:** Navigation lights must have a specific "Sector" (e.g., 112.5° for side lights) and a "Range of Visibility" (usually 2–6 nautical miles). 
*   **Chromaticity:** The colors (Red, Green, White) must meet strict IEC standards to ensure they are not confused with shore lights.

---

# Part 2 – Major Components & System Layout

## 2.1 Main Lighting Switchboards
Located in the Engine Control Room or Accommodation service spaces. They divide the load into "Zones" to prevent a single short circuit from plunging a whole deck into darkness.

## 2.2 Low Location Lighting (LLL)
A mandatory requirement for passenger ships and some tankers.
*   **Physics:** Photo-luminescent strips or low-power LEDs mounted near the floor. **Reason:** In a fire, smoke rises. High-level lighting becomes invisible, but LLL allows the crew to see the escape path below the smoke layer.

## 2.3 Navigation Light Controller
A dedicated panel on the Bridge with dual power feeds. 
*   **Function:** Monitors the "Filament" or "LED Chip" health. **Physics:** If a bulb burns out, the controller triggers an alarm and automatically switches to the "Reserve" bulb (if fitted).

## 2.4 Deck Floodlights (Ex-Proof)
Located on the open deck and in hazardous areas (Purifier room, Pump room).
*   **Physics:** Must be "Explosion-proof" (Ex-d or Ex-e) to prevent an internal electrical spark from igniting external gasses.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Illumination:** All escape routes are bright and clear.
*   **Insulation:** The ground-fault monitor for the lighting system shows "Clear" (Infinity).
*   **Status:** The Navigation Light panel shows all "Green" status.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **System Voltage** | 220 V ± 5% | < 190 V (Dimming/Flicker) |
| **Insulation Resist.** | > 1.0 MΩ | < 0.1 MΩ (Earth Fault) |
| **Emergency Battery** | 26.5 V – 27.5 V | < 23 V (Battery Low) |
| **Lux Level (Bridge)** | Dimmable to 0 | Non-dimmable (Night Blind)|

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Ghost" Earth Faults (DC System)
*   **Symptom:** Earth fault alarm on the 24V DC lighting circuit.
*   **Root Cause:** Moisture in the navigation lights on the mast or a corroded "Deck Service" plug.
*   **Expert Insight:** These faults often disappear when the sun comes out and reappear when it rains.

## 4.2 LED "Flicker"
*   **Symptom:** LED lamps strobe or flash rapidly.
*   **Root Cause:** Harmonic distortion from large VFDs or an aging "LED Driver" (AC/DC converter).
*   **Physics:** LEDs react instantly to voltage fluctuations. If the capacitor in the driver is failing, the 60Hz ripple becomes visible.

## 4.3 Nav-Light Controller Alarm (False)
*   **Symptom:** Alarm shows "Bulb Fail" but the light is on.
*   **Root Cause:** Wrong wattage bulb installed or high-resistance connection at the lamp socket.
*   **Physics:** The controller measures the "Current Draw." If the bulb is 40W but the controller expects 60W, it thinks the bulb is dying.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 Hunting an Earth Fault via "Zone Isolation"
*   **Trade Trick:** To find a lighting earth fault without killing the power, use a **Clamp-on Earth Leakage Meter**. **Method:** Clamp the meter around the entire cable (Positive and Negative together). **Physics:** If there is no fault, the currents cancel out to 0mA. If you see a reading (e.g., 50mA), that branch has an earth fault.

## 5.2 The "Incandescent Backup" Rule
*   **Expert Insight:** Some older Navigation Light controllers don't work with modern LED replacement bulbs because the LED current draw is too low. **Trade Trick:** If you must use LEDs, you may need to add a **"Load Resistor"** in parallel to trick the controller into thinking a standard bulb is present.

## 5.3 Quick Battery "Load Test"
*   **Trade Trick:** To test the emergency 24V batteries, turn off the main charger and turn on all emergency lights. **Measure the voltage drop after 5 minutes.** If it drops below 24.0V, the batteries are "Sulfated" and will not last the required 30 minutes in a blackout.

## 5.4 Cleaning "Salt-Caked" Lenses
*   **Trade Trick:** Navigation light lenses on the mast get covered in salt spray, reducing their range. **Method:** Wash with warm fresh water and a dash of vinegar. **Warning:** Never use abrasive pads on plastic lenses; the scratches will "diffuse" the light beam and make the ship look further away than it is.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Nav-Light Test:** Test all navigation lights from the Bridge panel (Main and Reserve).
*   **Indicator Check:** Verify the operation of all "Exit" signs and escape lighting.

## 6.2 Quarterly Battery Maintenance
*   **Method:** Clean battery terminals and apply petroleum jelly. Check the electrolyte level.

## 6.3 Annual Lighting Survey
*   **Requirement:** Verify that the "Emergency Lighting" automatically illuminates upon loss of main power. **Test:** Kill the main lighting breaker for a specific zone and ensure the emergency lights fire up instantly.

---

# Part 7 – Miscellaneous Knowledge

*   **Helideck Lighting:** Specialized "Green" perimeter lights and "Yellow" floodlights. They must be dimmable to avoid blinding the pilot during landing.
*   **Bridge Dimming:** All bridge equipment (ECDIS, Radar, Compass) must have a centralized "Dimming" control. If one screen is too bright at night, the officer loses their **Night Vision** (Physics: Rhodopsin bleach in the eye).

**End of Document**
