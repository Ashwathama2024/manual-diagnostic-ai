# MARINE BALLAST WATER TREATMENT SYSTEM (BWTS) – COMPLETE CORE KNOWLEDGE

**Equipment:** IMO/USCG Approved Ballast Water Management System (e.g., Alfa Laval PureBallast, Panasia GloEn-Patrol, Erma First FIT, Techcross ECS)

**Folder Name:** ballast_water_treatment

**Prepared by:** Senior Marine Engineer & Environmental Compliance Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Water Treatment

## 1.1 The Biology of Invasive Species (D-2 Standard)
Ballast water transfer is the primary vector for the global spread of harmful aquatic organisms (e.g., Zebra Mussels, Cholera, Toxic Algae).
*   **The Physics of Compliance:** The IMO D-2 standard mandates that discharged ballast water must contain:
    *   < 10 viable organisms per $m^3$ (> 50 microns).
    *   < 10 viable organisms per $ml$ (10–50 microns).
    *   Specific limits for Indicator Microbes (E. coli, etc.).

## 1.2 The Physics of UV Sterilization
*   **The Physics:** High-intensity Ultraviolet (UV-C) light at 254nm wavelength penetrates the cell walls of organisms.
*   **DNA Damage:** The UV light breaks the DNA/RNA bonds, rendering the organisms "Inviable" (unable to reproduce). It doesn't necessarily "kill" them instantly, but it stops the infestation.
*   **UV Transmittance (UVT):** Treatment efficiency depends on the clarity of the water. Muddy or "Tea-colored" water absorbs UV light, requiring the system to slow down the flow to maintain the required "Dose."

## 1.3 Electro-chlorination (Electrolysis) Physics
*   **The Physics:** Seawater ($NaCl + H_2O$) is passed through an electrolytic cell.
*   **The Reaction:** Electricity converts the salt into **Sodium Hypochlorite** ($NaOCl$) and **Hydrogen Gas** ($H_2$). 
*   **Disinfection:** The hypochlorite acts as a powerful oxidant, killing all organic life in the tanks. **Warning:** Hydrogen gas is explosive and must be vented safely.

---

# Part 2 – Major Components & System Layout

## 2.1 The Automatic Filter (Stage 1)
Usually a 20–40 micron mesh filter.
*   **Function:** Removes larger organisms, sand, and silt. This reduces the "load" on the UV/Chemical stage and prevents organisms from "hiding" behind silt particles.
*   **Back-flushing:** Automatically triggers based on $\Delta P$.

## 2.2 The Treatment Unit (Stage 2)
*   **UV Reactor:** A stainless steel chamber containing multiple high-power UV lamps inside quartz sleeves.
*   **Electrolysis Cell:** A series of titanium plates coated with precious metals (Iridium/Ruthenium).

## 2.3 TRO and Salinity Sensors
*   **TRO Sensor:** Measures "Total Residual Oxidant" (Chlorine) in the water.
*   **Salinity Sensor:** Electrolysis requires salt. If the ship is in a freshwater river, the system may need to "Inject" brine (concentrated saltwater) to work.

## 2.4 Neutralization (De-chlorination) Unit
*   **Function:** Before discharge, any remaining chlorine must be neutralized using **Sodium Bisulfite**.
*   **Physics:** This ensures the discharged water is non-toxic to the local harbor environment.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **UV Intensity:** Stable and above the minimum "Type Approval" limit.
*   **Flow Control:** The system automatically throttles the ballast pump to match the water clarity.
*   **Data Logging:** The BWTS computer is recording GPS, Time, and Treatment parameters.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **UV Intensity** | 400 – 1000 W/m² | < 300 W/m² (Slow Flow) |
| **Filter ΔP** | 0.1 – 0.3 bar | > 0.6 bar (Back-flush) |
| **TRO Level (In-tank)**| 2.0 – 5.0 mg/l | > 8.0 mg/l (Over-dosing) |
| **TRO Level (Out)** | < 0.1 mg/l | > 0.2 mg/l (Neutralize Fail) |
| **Hydrogen Level** | < 1% LEL | > 2% (Ventilation Fail) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Filter Plugging (The "Brown Water" Problem)
*   **Symptom:** Constant back-flushing; ballast flow drops to near zero.
*   **Root Cause:** Heavy silt or algae bloom in the harbor water.
*   **Solution:** Bypass the system? **NO.** This is illegal. You must ballast at a slower rate or use the "Internal Circulation" mode if fitted.

## 4.2 UV Lamp Failure
*   **Symptom:** "Lamp Fail" alarm; UV intensity drops.
*   **Root Cause:** Lamp reach its end-of-life (approx. 1000–2000 hours) or the quartz sleeve is fouled with scale.

## 4.3 TRO Sensor "Drift"
*   **Symptom:** Chlorine levels showing 0.0 even when the electrolysis cell is working.
*   **Root Cause:** Blocked reagent pump or fouled measuring electrodes.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 Cleaning Quartz Sleeves "In-Situ"
*   **Trade Trick:** Many UV reactors have automatic mechanical "Wipers." If the intensity is low, the wipers are likely stuck. **Expert Insight:** Perform a "CIP" (Cleaning In Place) using a 5% Citric Acid solution to dissolve the calcium scale that the wipers can't reach.

## 5.2 The "Salinity Boost" for Electrolysis
*   **Expert Insight:** If you are in a river (low salinity) and the system won't start, check if you have a **Seawater Header Tank**. **Trade Trick:** You can sometimes "Pre-fill" the treatment cell with high-salinity water from the header tank to get the electrolysis reaction started.

## 5.3 Manual Reagent Check
*   **Trade Trick:** TRO sensors use chemicals (DPD). If the reading is erratic, check the reagent bottles. **If the chemical has turned "Dark Pink" in the bottle, it is expired/oxidized.** Replace with fresh reagents to restore accuracy.

## 5.4 Detecting "Illegal" Treatment
*   **Expert Insight:** PSC inspectors will look for evidence of "Treatment Bypass." **Trade Trick:** Check the software log for **"GPS Signal Lost"** or **"Flow Meter Fault"** during ballasting. These are red flags that the system was being manipulated.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Monthly Routine
*   **Leak Check:** Inspect the UV reactor gaskets. UV light degrades rubber quickly.
*   **Calibration:** Verify the TRO sensor against a handheld colorimeter.

## 6.2 Annual Sensor Verification
*   **Requirement:** The UV intensity sensors and TRO meters must be calibrated and certified by an authorized technician every 12 months to maintain the "International BWM Certificate."

## 6.3 Quartz Sleeve Replacement
*   **Method:** Replace sleeves every 2 years. They become brittle and "Solarized" (turned opaque) by the constant UV exposure, reducing the effective dose.

---

# Part 7 – Miscellaneous Knowledge

*   **Treatment at Intake vs. Discharge:** UV systems treat water **both ways** (Intake and Discharge). Electrolysis systems usually treat **only at Intake** and then neutralize at Discharge.
*   **USCG vs. IMO:** US Coast Guard rules are stricter than IMO. Ensure your BWTS is operating in the correct "USCG Mode" when entering US waters.

**End of Document**
