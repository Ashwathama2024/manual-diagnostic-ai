# MARINE SPEED LOG – COMPLETE CORE KNOWLEDGE

**Equipment:** Speed Measurement System (Doppler Log or Electromagnetic Log) - (e.g., Furuno DS-80, JRC JLN-740, Sperry Marine NAVIKNOT, Skipper EML224)

**Folder Name:** speed_log

**Prepared by:** Expert Marine Electronics & Navigation Systems Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Speed Measurement

## 1.1 The Physics of STW vs. SOG
A ship has two different speeds that are equally important for navigation:
1.  **Speed Through Water (STW):** Measured relative to the water surrounding the hull. **Physics:** Essential for calculating "Engine Slip" and for Radar/ARPA collision avoidance (it shows the ship's physical heading through the medium).
2.  **Speed Over Ground (SOG):** Measured relative to the Earth's surface. **Physics:** Used for true position fixing and cross-track error calculation.

## 1.2 Doppler Effect Physics (Doppler Log)
*   **The Physics:** $f_d = \frac{2 \cdot v \cdot f_s \cdot \cos(\theta)}{c}$. 
    Where $f_d$ is the frequency shift, $v$ is the ship's speed, and $f_s$ is the source frequency. 
*   **The Process:** The log emits a pulse of sound. The "Echo" from the water molecules or the seabed is shifted in frequency proportional to the ship's velocity.
*   **Water Track vs. Bottom Track:** 
    *   **Water Track:** Echo comes from plankton/bubbles in the water (Measures STW).
    *   **Bottom Track:** Echo comes from the seabed (Measures SOG). Bottom track only works in depths < 200m.

## 1.3 Electromagnetic (EM) Induction Physics
*   **The Physics:** Faraday's Law of Induction. $V = B \cdot L \cdot v$. 
*   **The Process:** A coil inside the transducer creates a magnetic field $(B)$. As the conductive seawater $(L)$ flows past the hull at velocity $(v)$, a tiny voltage $(V)$ is induced in the electrodes. This voltage is directly proportional to the ship's speed through the water.

---

# Part 2 – Major Components & System Layout

## 2.1 The Transducer (The Sensor)
Mounted flush with the hull in the forward section. 
*   **Doppler Transducer:** Contains ultrasonic crystals.
*   **EM Transducer:** Contains an electromagnet and sensing electrodes.

## 2.2 The Sea Valve (Gate Valve)
A heavy-duty valve that allows the transducer to be removed for cleaning or replacement while the ship is afloat. **Physics:** Failure of this valve can lead to an engine room flood.

## 2.3 The Electronic Unit (Processor)
Amplifies the tiny micro-volt signals from the transducer and converts them into digital speed data (NMEA 0183).

## 2.4 Displays and Repeaters
Digital speed indicators on the Bridge, Bridge Wings, and sometimes in the Engine Control Room.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Stability:** The speed reading is steady and doesn't "Jump" (e.g., from 15.0 to 18.0 knots instantly).
*   **Correlation:** STW and SOG are similar in areas with no current.
*   **Transducer Current:** Stable mA draw for the EM coil.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Speed Range** | -5 to +40 knots | Varies by vessel |
| **Tracking Mode** | Bottom (Shallow) / Water (Deep) | Mode Lost (No Fix) |
| **Calibration Constant**| User Defined (e.g., 1.05) | Sudden Change (Fouling) |
| **Signal Strength** | > 60% | < 20% (Weak Signal) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Boundary Layer Effect (Inaccuracy)
*   **Symptom:** Speed log reads lower than actual speed (GPS speed) even with no current.
*   **Root Cause:** The transducer is located in the "Boundary Layer" (water that is being dragged along by the hull friction).
*   **Physics:** As the hull fouls with barnacles, the boundary layer thickens, reducing the measured water velocity.

## 4.2 "No Signal" / Zero Speed
*   **Symptom:** Speed display shows "0.0" while the ship is moving.
*   **Root Cause:** Transducer electrodes are fouled with oil or paint, or the Doppler crystal is cracked.
*   **Expert Insight:** Often happens after bunkering if oil spills and coats the hull sensors.

## 4.3 Interference (Acoustic Noise)
*   **Symptom:** Speed reading fluctuates wildly.
*   **Root Cause:** "Cross-talk" from the Echo Sounder or noise from the Bow Thruster.
*   **Physics:** The log and echo sounder operate on similar frequencies. They must be synchronized (Triggered) to avoid interference.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Cleaning" Routine (EM Log)
*   **Trade Trick:** If the EM log is erratic, the electrodes are likely oily. **Method:** If you cannot drydock, use a "Diver's Brush" to clean the transducer face. **Expert Insight:** Even a thin film of "Marine Growth" (Slime) creates an insulating layer that blocks the induced voltage.

## 5.2 Calibrating "Sea Trial" Style
*   **Trade Trick:** To calibrate the log, perform a **"Measured Mile"** run. 
*   **Method:** Run the ship at a steady RPM between two GPS waypoints in both directions (to cancel out current). Compare the average GPS speed to the Log speed. **Adjust the "Gain" or "Log Constant" in the menu until they match.**

## 5.3 Detecting a Failing Doppler Crystal
*   **Trade Trick:** Listen to the transducer using a plastic tube. **Expert Insight:** You should hear a faint "Ticking" sound if the transmitter is working. If it is silent, the internal crystal has failed or the cable is snapped.

## 5.4 Using "GPS Speed" as a Backup
*   **Expert Insight:** If the speed log fails, most ECDIS units can use GPS speed (SOG) for navigation. **Warning:** You MUST switch the ARPA to **"Sea Stabilized"** mode manually. If the ARPA uses SOG for collision avoidance, the "Target Vectors" will be wrong in areas with strong currents.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Comparison:** Compare Log Speed vs. GPS Speed and Engine RPM.
*   **Drain Check:** Open the sea valve cofferdam drain to ensure no water is leaking into the transducer housing.

## 6.2 Annual Performance Test (APT)
*   **Requirement:** Verify the NMEA data output and the sensor calibration. 

## 6.3 Transducer Protection (Drydock)
*   **Critical Action:** During hull painting, **DO NOT PAINT the speed log transducer.** Paint acts as an insulator for EM logs and a sound absorber for Doppler logs. **Trade Trick:** Cover the transducer with a greased plastic cup during painting and remove it at the last moment.

---

# Part 7 – Miscellaneous Knowledge

*   **Docking Log:** High-precision Doppler logs used on VLCCs and Gas Carriers. They measure **Fore/Aft** and **Port/Stbd** speed at the bow and stern to help the Pilot during berthing.
*   **NMEA VBW Telegram:** The standard data format for speed logs. It includes "Water Speed" and "Ground Speed" in a single message.

**End of Document**
