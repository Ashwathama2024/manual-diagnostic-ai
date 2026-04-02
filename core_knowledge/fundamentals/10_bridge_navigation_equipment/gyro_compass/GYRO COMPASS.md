# MARINE GYROCOMPASS – COMPLETE CORE KNOWLEDGE

**Equipment:** True North Seeking Gyrocompass (e.g., Sperry Marine MK-37, Anschütz Standard 22, Simrad GC80, Yokogawa CMZ900)

**Folder Name:** gyro_compass

**Prepared by:** Senior Marine Electronics & Navigation Engineer (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Gyroscopes

## 1.1 The Physics of Rigidity in Space
A gyrocompass relies on a rapidly spinning wheel (Rotor) that exhibits "Rigidity in Space."
*   **The Physics:** According to Newton's First Law, the spinning rotor wants to keep its axis pointed at a fixed point in the universe (e.g., a distant star), regardless of how the ship moves around it.

## 1.2 The Physics of Precession
*   **The Problem:** Rigidity in space is not enough to find North. As the Earth rotates, the gyro's axis will seem to "tilt" up and away from North.
*   **The Physics Solution:** **Precession**. When a force (Torque) is applied to a spinning gyro, it moves at a 90° angle to that force.
*   **North-Seeking:** By adding a weight (Gravity Control) to the gyro, the Earth's rotation creates a torque that forces the gyro axis to precess until it aligns perfectly with the Earth's rotational axis (True North).

## 1.3 Correction Physics (Latitude and Speed Error)
A gyrocompass is affected by the ship's own movement.
*   **The Physics:** The gyro sees the "Vector Sum" of the Earth's rotation ($\approx 900$ knots at the equator) and the ship's speed. 
*   **Result:** This creates a small error (Steaming Error). The ECDIS/Gyro computer must calculate this error based on the ship's **Latitude** and **Speed** and apply a mathematical correction to the heading.

---

# Part 2 – Major Components & System Layout

## 2.1 The Gyro Element (Sensitive Element)
*   **The Rotor:** Spins at high speed (e.g., 12,000 – 20,000 RPM) inside a vacuum-sealed "Sphere" or "can."
*   **Suspension:** The sphere is suspended in a fluid (Silicone Oil) or by a thin wire (Torsion wire) to eliminate friction.

## 2.2 The Follow-up System
*   **Function:** As the ship turns, the sensitive element stays pointed North. The "Follow-up" frame must rotate to match the ship's heading without touching the sensitive element.
*   **Physics:** Uses "Phantom Ring" technology with capacitive or optical sensors to maintain a zero-friction gap between the gyro and the ship.

## 2.3 The Master Compass and Repeaters
*   **Master Compass:** The unit containing the sensitive element, usually located in a quiet, low-vibration room near the ship's centerline.
*   **Repeaters:** Digital or analog displays on the bridge wings, steering stand, and emergency steering position.

## 2.4 Interface (NMEA 0183 / Step)
The Gyro sends heading data to the Radar, ECDIS, AIS, and Autopilot. **Physics:** Any "Lag" in this data will cause the Autopilot to "S-turn" or the Radar to smear.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Settling Time:** After starting from cold, the gyro takes **3 to 5 hours** to "Settle" on North.
*   **Repeatability:** The repeaters match the master compass within 0.1°.
*   **Stability:** The heading does not fluctuate while the ship is at a steady course in calm water.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Rotor Speed** | 12,000 – 19,000 RPM | Low RPM (Bearing Fail) |
| **Supporting Fluid Temp**| 45°C – 55°C | > 65°C (High Temp Trip) |
| **Follow-up Current** | Stable mA | Oscillating (System Hunting)|
| **Settled Error** | < 0.5° | > 1.0° (Maintenance Req.) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Gyro Drift" (Unstable Heading)
*   **Symptom:** The heading slowly changes by 2-3° even when the ship is stationary.
*   **Root Cause:** Improper Latitude or Speed correction.
*   **Expert Insight:** If the GPS signal to the gyro is lost, the gyro uses the "Last Known Speed." If you are in port (0 knots) but the gyro thinks you are at 20 knots, it will drift.

## 4.2 "Step-Fail" Alarm
*   **Symptom:** Bridge repeaters are "Frozen" or show a different heading than the master.
*   **Root Cause:** Failed serial data cable or a "Short Circuit" in one of the repeater motors.

## 4.3 High Vibration / Noise
*   **Symptom:** A high-pitched "Scream" or "Growl" from the master unit.
*   **Root Cause:** Failing rotor bearings.
*   **Physics:** Because the rotor spins at 20,000 RPM, a tiny bearing defect will generate intense heat and noise. Failure is imminent.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Settle" Test
*   **Trade Trick:** If you suspect the gyro is inaccurate, perform a **"Transit Bearing"** check. Align the ship so two known landmarks (e.g., two lighthouses) are in a straight line. Compare the chart bearing to your gyro heading. **Error > 1° means the gyro needs professional calibration.**

## 5.2 Cleaning "Slip Rings"
*   **Trade Trick:** If the follow-up system is "Hunting" (moving back and forth), the electrical slip rings are likely dirty. **Expert Insight:** Clean the rings with a specialized "Burnishing Tool" or a lint-free cloth with electronic cleaner. **Never use sandpaper!**

## 5.3 The "Bubble" Check (Fluid Type)
*   **Trade Trick:** Look through the sight glass of the master unit. **If you see air bubbles in the supporting fluid**, the damping will be erratic. You must add degassed silicone oil using a syringe.

## 5.4 Starting in "Fast Settle" Mode
*   **Expert Insight:** Some gyros have a "Fast Settle" button. **Trade Trick:** Only use this if the ship is perfectly stationary at the pier. If you use it while the ship is moving or rolling, the gyro will "precess" to a false North and stay there for hours.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Annual Service (Mandatory)
*   **Requirement:** The gyro must be serviced by an authorized technician every 12 months.
*   **Tasks:** Replace the sensitive element (if required), change the supporting fluid, and replace the filters and gaskets.

## 6.2 Quarterly Battery Check
*   **Check:** The Gyro must have a dedicated backup battery. Ensure the battery can maintain the heading for at least **30 minutes** during a blackout. If the gyro "Tumbles" during a blackout, you will be blind for 4 hours while it re-settles.

## 6.3 Repeater Synchronization
*   **Method:** Every week, verify that the Bridge repeaters match the Master. If they are off, use the "Align" function to sync them digitally.

---

# Part 7 – Miscellaneous Knowledge

*   **Tumbling:** If the gyro's limits of pitch/roll are exceeded (e.g., in a severe storm), the gyro "Tumbles." The sensitive element hits the frame and loses its North-seeking ability. It must be restarted.
*   **Fiber Optic Gyro (FOG):** Modern alternative. It has **No Moving Parts**. It uses the "Sagnac Effect" (interference of laser light) to measure rotation. Much more reliable but more expensive.

**End of Document**
