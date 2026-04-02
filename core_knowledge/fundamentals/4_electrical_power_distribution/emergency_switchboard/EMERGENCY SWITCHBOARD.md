# MARINE EMERGENCY SWITCHBOARD (ESB) – COMPLETE CORE KNOWLEDGE

**Equipment:** Emergency Power Distribution Switchboard

**Folder Name:** emergency_switchboard

**Prepared by:** Senior Marine Electrical Engineer & Safety Systems Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Emergency Power

## 1.1 The Physics of Independence (Separation)
Like the emergency fire pump, the ESB must remain functional if the main Engine Room is lost to fire or flooding.
*   **The Physics of Location:** It is located above the "Bulkhead Deck" (usually in the same compartment as the Emergency Generator).
*   **Interconnector Physics:** Under normal conditions, the ESB is powered by the Main Switchboard (MSB) via a dedicated **Interconnector Cable**. 
*   **Non-Return Logic:** The interconnector is fitted with a breaker that only allows power to flow from the **MSB to the ESB**. In a blackout, this breaker trips to prevent the Emergency Generator from trying to power the entire ship (which would immediately overload and trip the emergency set).

## 1.2 Automatic Mains Failure (AMF) Physics
*   **The Relay:** An AMF relay (e.g., Deep Sea, ComAp, or Deif) constantly monitors the voltage of the MSB. 
*   **The Physics of Detection:** If the MSB voltage drops below 80% for more than 0.5 seconds, the relay initiates the "Blackout Sequence":
    1.  Trip the Interconnector Breaker.
    2.  Send a start signal to the Emergency Generator.
    3.  Once the generator reaches 95% voltage/frequency, close the Emergency Generator Breaker.

## 1.3 The "Dead-Ship" Start Physics
The ESB is the "First Responder" in a dead-ship situation (total loss of all power, including compressed air). It must provide the power for the small **Emergency Air Compressor** to refill the start-air bottles for the main generators.

---

# Part 2 – Major Components & Emergency Loads

## 2.1 The ESB Busbars
Usually smaller than MSB busbars but must be rated for the full output of the Emergency Generator plus the MSB interconnector capacity.

## 2.2 Emergency Load Distribution
Only critical "Life Safety" and "Navigation" loads are connected to the ESB:
*   **Navigation Bridge:** Radar, GPS, ECDIS, GMDSS (Radio).
*   **Emergency Lighting:** Escape routes, lifeboat stations, and Engine Room critical areas.
*   **Safety Systems:** Fire detection, internal communications, and General Alarm.
*   **Machinery:** Emergency Fire Pump, Steering Gear (one motor), and Emergency Air Compressor.

## 2.3 Battery Charging Section
The ESB contains the chargers for all critical 24V DC systems:
*   **Radio Battery:** Supplies the GMDSS station for 1–6 hours.
*   **Emergency Gen Start Battery:** Two independent sets of batteries.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Interconnector:** Breaker is CLOSED (ESB is being "fed" by MSB).
*   **Ready for Start:** Emergency Generator selector switch is in "AUTO."
*   **Battery Status:** All chargers show "Float" charge and 100% capacity.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **System Voltage** | 440 V / 220 V | ± 10% Deviation |
| **System Frequency** | 60 Hz | ± 2% Deviation |
| **Battery Voltage** | 26.5 V – 27.5 V | < 23 V (Battery Low) |
| **Insulation Resist.** | > 1.0 MΩ | < 0.1 MΩ (Earth Fault) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Failed to Start" during Blackout
*   **Symptom:** MSB goes dark, but the Emergency Generator never fires.
*   **Root Cause:** Dead starting batteries or a faulty "Blackout Relay" in the MSB that failed to send the signal.

## 4.2 Interconnector Trip (Nuisance)
*   **Symptom:** ESB loses power while the MSB is healthy.
*   **Root Cause:** Over-current on the ESB (too many loads turned on) or a "Reverse Power" trip due to a faulty voltage sensor.

## 4.3 Earth Faults on DC Circuits
*   **Symptom:** "DC Earth Fault" alarm on the ESB.
*   **Root Cause:** Moisture in the navigation lights on the mast or a ground fault in the fire detection loop.
*   **Expert Insight:** DC earth faults are notoriously hard to find. They often appear only in rainy weather.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Blackout Simulation" Test
*   **Trade Trick:** Once a week, test the system by **manually tripping the MSB interconnector breaker** (with the Captain's permission!). This forces the AMF relay to see a "real" blackout and start the generator. **A successful test proves the entire sequence works.**

## 5.2 The "Preferred Trip" Logic
*   **Expert Insight:** If the Emergency Generator is running and the load increases too much, the ESB will perform a "Preferred Trip." It will shut down non-essential emergency loads (like some ventilation) to keep the Bridge and Steering Gear alive.

## 5.3 Checking the "Back-Feed" Interlock
*   **Trade Trick:** Ensure the mechanical interlock between the Interconnector and the Generator Breaker is free. **They must NEVER be closed at the same time.** If they were, the Emergency Gen would try to "back-feed" into a dead engine room, resulting in a catastrophic explosion of the generator.

## 5.4 Using the "Manual Parallel" for Testing
*   **Trade Trick:** Some ESBs allow you to "Parallel" the Emergency Generator with the MSB for testing purposes (load testing). **Only do this if you are a skilled electrician**, as the Emergency Generator governor is often slower and harder to synchronize than the main sets.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Battery Electrolyte:** Check levels in lead-acid batteries. Top up with distilled water.
*   **Start Test:** Verify the generator starts and reaches voltage within 45 seconds (SOLAS limit).

## 6.2 Annual Breaker Service
*   **Method:** Inspect the arc chutes and main contacts of the Emergency Gen Breaker. Since it is rarely used under load, the mechanism can "seize." **Apply a small amount of aerosol lubricant to the trip linkages.**

## 6.3 Cleaning the Battery Locker
*   **Method:** Ensure the battery locker ventilation is clear. Batteries produce **Hydrogen Gas** during charging, which is highly explosive. A blocked vent + a spark = disaster.

---

# Part 7 – Miscellaneous Knowledge

*   **Temporary Emergency Power:** For some ships, a large UPS (Battery bank) provides the "Transition" power for the 45 seconds it takes the emergency generator to start. This ensures the Bridge never goes dark.
*   **Emergency Gen "Exercise"**: Don't just start the engine; **apply at least 50% load** for 30 minutes once a month to prevent "Wet-Stacking" (unburned fuel in the exhaust).

**End of Document**
