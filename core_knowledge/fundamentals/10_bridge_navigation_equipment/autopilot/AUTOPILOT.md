# MARINE AUTOPILOT – COMPLETE CORE KNOWLEDGE

**Equipment:** Automatic Steering Control System (e.g., Sperry Marine NAVIPILOT, Anschütz PilotStar, Raytheon Anschütz NautoPilot, Simrad AP series)

**Folder Name:** autopilot

**Prepared by:** Expert Marine Systems & Automation Engineer (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Automatic Steering

## 1.1 The Physics of Heading Control (The PID Loop)
The Autopilot is a "Closed-Loop" control system. It compares the "Actual Heading" from the Gyro with the "Set Heading" from the officer.
*   **The Physics:** It uses a Proportional-Integral-Derivative (PID) algorithm:
    1.  **Proportional (P):** The rudder angle is proportional to the heading error. (Large error = Large rudder).
    2.  **Integral (I):** Compensates for "Steady-State Error" caused by wind or current pushing the ship off course. It "remembers" the drift and adds a constant offset.
    3.  **Derivative (D):** Reacts to the **Rate of Turn**. If the ship is swinging fast, the D-term applies "Counter-rudder" to stop the swing before it overshoots the set course.

## 1.2 Adaptive Steering Physics
Modern autopilots are "Adaptive." 
*   **The Physics:** The system automatically calculates the ship's characteristics (Inertia, Rudder Authority) based on how the ship reacts to steering commands.
*   **Weather Adjustment:** In heavy seas, the ship "yawns" naturally. An adaptive autopilot ignores this high-frequency movement to avoid "hunting" the rudder, which saves fuel and reduces wear on the steering gear.

## 1.3 Speed and Draft Influence
*   **The Physics:** Rudder effectiveness is proportional to the square of ship speed ($V^2$). At high speed, only 2° of rudder is needed for a turn. At low speed, 20° might be required. The autopilot must receive a **Speed Log** signal to adjust its "Gain."

---

# Part 2 – Major Components & System Layout

## 2.1 The Control Unit (The Interface)
The panel on the bridge where the officer sets the course, chooses the mode (Auto/Manual/Track), and adjusts the steering parameters.

## 2.2 The Autopilot Processor (The Brain)
Often a dual-redundant PLC or industrial computer. It performs the PID calculations 10–50 times per second.

## 2.3 Feedback Unit (Rudder Angle Transmitter)
A high-precision sensor mounted on the rudder stock. **Physics:** It tells the autopilot exactly where the rudder is. If this feedback is "noisy" or has a "dead-band," the ship will "S-turn" (Oscillate).

## 2.4 Interface (Gyro, Log, and ECDIS)
*   **Track Mode:** The autopilot receives "Cross-Track Error" (XTE) from the ECDIS and automatically steers the ship to stay on the charted line.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Course Keeping:** In calm water, the ship stays within ±0.5° of the set course.
*   **Rudder Activity:** The rudder moves in small, smooth increments. Constant "Hard-over to Hard-over" movement indicates a parameter error.
*   **Transition:** Switching from "Auto" to "Manual" (Hand steering) is instantaneous and seamless.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Heading Deviation** | ± 1.0° | > 5.0° (Off-Course Alarm) |
| **Rudder Limit** | 5° – 15° (at sea) | User Defined |
| **Rate of Turn** | 10° – 30° / min | High Rate (Safety Risk) |
| **Yaw Setting** | 1 (Calm) – 5 (Rough) | Varies by weather |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "S-Turning" (Oscillation)
*   **Symptom:** The ship constantly swings left and right of the course.
*   **Root Cause:** "Gain" (P-term) is set too high, or the "Weather" (Yaw) setting is too low.
*   **Expert Insight:** Often happens if the **Speed Log** signal is lost and the Autopilot defaults to "High Speed" mode while the ship is actually slow.

## 4.2 "Off-Course" Alarm
*   **Symptom:** Alarm triggers even though the ship looks like it's on course.
*   **Root Cause:** The Gyrocompass is drifting, or there is a discrepancy between the Master Gyro and the Autopilot input.

## 4.3 "Feedback Fail"
*   **Symptom:** Autopilot trips to "Manual" mode instantly.
*   **Root Cause:** The Rudder Feedback Linkage is loose or the potentiometer is worn.
*   **Danger:** Without feedback, the autopilot doesn't know where the rudder is and could drive the ship into a dangerous turn.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Rudder Limit" Safety
*   **Trade Trick:** Always set the "Rudder Limit" to approx. 10° – 15° when in the open sea. **Expert Insight:** If the Autopilot has a software glitch, the rudder limit prevents it from going to "Hard-over," which could shift the cargo or capsize a small vessel at high speed.

## 5.2 Tuning the "Weather" Setting
*   **Trade Trick:** If the steering gear pumps are running constantly in heavy weather, **increase the Yaw/Weather setting**. This creates a "Dead-zone" around the course. The ship will swing more, but you will save the steering gear from overheating and reduce fuel consumption.

## 5.3 Detecting "NMEA Lag"
*   **Expert Insight:** If the ship reacts slowly to course changes, the serial data from the Gyro might be "Lagging" due to a slow baud rate or network congestion. **Trade Trick:** Check the "Heading Update Rate" in the diagnostic menu. It should be at least **10 Hz** (10 updates per second).

## 5.4 The "Manual Override" Test
*   **Critical Action:** Before every departure, test the **Override** function. While in "Auto," moving the hand-steering wheel should either trip the system to "Manual" or take immediate control. **A stuck "Auto" button is a grounding risk.**

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Routine
*   **Sync Check:** Verify the Autopilot heading matches the Master Gyro.
*   **Parameters:** Adjust the Yaw/Weather setting based on the current sea state.

## 6.2 Annual Performance Test (APT)
*   **Requirement:** An authorized technician must check the PID tuning and the interface integrity.

## 6.3 Feedback Linkage Inspection
*   **Check:** Go to the steering gear room every 3 months. Check the ball joints on the rudder feedback rod. If there is "Slop" (Play) in the joints, the Autopilot will never be able to steer a straight line.

---

# Part 7 – Miscellaneous Knowledge

*   **NFU (Non-Follow Up):** An emergency steering mode where the officer uses a "Tiller" to move the rudder directly. The rudder stays where you leave it. Autopilots usually have an NFU backup.
*   **HSC (High Speed Craft):** Specialized autopilots for fast ferries that react much faster to prevent "Broaching" in a following sea.

**End of Document**
