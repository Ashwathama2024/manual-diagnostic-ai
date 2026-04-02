# MARINE PROVISION REFRIGERATION – COMPLETE CORE KNOWLEDGE

**Equipment:** Ship's Food Store Refrigeration Plant (e.g., Bitzer, Bock, York, Sabroe)

**Folder Name:** provision_refrigeration

**Prepared by:** Senior Marine Engineer & HVAC Specialist (25+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Food Preservation

## 1.1 The Physics of Latent Heat of Fusion
The provision plant must keep meat and fish below $-18^\circ C$.
*   **The Physics:** Freezing a product requires removing the **Latent Heat of Fusion**. Even if the room air is cold, the "core" of a piece of meat remains at $0^\circ C$ until all water molecules have crystallized into ice.
*   **Thermal Inertia:** Large quantities of food act as a "Thermal Battery." Once the stores are frozen, the compressor runs less frequently.

## 1.2 The Physics of Sublimation (Freezer Burn)
*   **The Problem:** In a freezer, ice can turn directly from a solid to a gas (Sublimation) if the air is too dry.
*   **Result:** This dries out the food ("Freezer Burn"). The provision plant must maintain a balance between cold air and minimal air velocity to reduce this effect.

## 1.3 Multi-Temperature Zone Physics
A single compressor set usually handles three different temperatures:
1.  **Meat/Fish Room:** $-18^\circ C$ to $-25^\circ C$.
2.  **Vegetable Room:** $+4^\circ C$ to $+6^\circ C$.
3.  **Lobby/Dry Store:** $+10^\circ C$ to $+15^\circ C$.
*   **The Physics:** This is achieved using **Evaporator Pressure Regulators (EPR)** or "Back-pressure Valves." These valves keep the pressure (and thus the temperature) higher in the vegetable room evaporator while the meat room is at a much lower pressure.

---

# Part 2 – Major Components & System Layout

## 2.1 Compressor Set
Usually small reciprocating compressors (often 2 or 3 units for redundancy).
*   **Oil Separator:** Vital for provision plants because the evaporators are far away and cold. Oil tends to "trap" in the cold coils and must be returned to the compressor.

## 2.2 Evaporators (Cooling Coils)
Located inside each room. They include fans to circulate air and **Electric Defrost Heaters**.

## 2.3 Solenoid Valves and Thermostats
Each room has its own thermostat. 
*   **Logic:** When the room is cold enough, the thermostat closes the **Liquid Line Solenoid Valve**. The compressor then "Pumps Down" and stops when the suction pressure drops.

## 2.4 Defrost Control Timer
*   **Physics:** Moisture from food and air-ingress (opening doors) freezes on the evaporator fins. This ice acts as an insulator ($U \downarrow$).
*   **The Cycle:** Every 4–8 hours, the cooling stops and electric heaters melt the ice. **Safety:** A "Drain Line Heater" ensures the melt-water doesn't re-freeze and block the drain pipe.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Room Temps:** Steady and within 1°C of setpoint.
*   **Evaporator Fins:** Clean and clear of heavy ice buildup.
*   **Sight Glass:** Clear liquid during the run cycle.

## 3.2 Typical Operating Parameters (R404a)

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Meat Room Temp** | -18°C to -22°C | > -15°C (Food Spoilage) |
| **Veg Room Temp** | +4°C to +6°C | < 0°C (Freezing Risk) |
| **Suction Pressure** | 1.0 – 2.0 bar | < 0.5 bar (Vacuum Risk) |
| **Discharge Pressure** | 12 – 16 bar | > 20 bar (Cooling Fail) |
| **Oil Level** | 1/2 Sight Glass | < 1/4 (Oil Trapped in Coils) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 "Short-Cycling"
*   **Symptom:** Compressor starts and stops every 2 minutes.
*   **Root Cause:** Low refrigerant charge or a faulty LP (Low Pressure) switch.
*   **Physics:** The compressor "Pumps Down" too quickly because there isn't enough gas to sustain the suction pressure.

## 4.2 Evaporator "Iced Solid"
*   **Symptom:** Room temperature rises; fans are running but no air is moving.
*   **Root Cause:** Failed defrost heater or a blocked drain line.
*   **Expert Insight:** If the "Drain Heater" fails, the ice builds from the bottom up, eventually swallowing the whole coil.

## 4.3 Food Wilting (Veg Room)
*   **Symptom:** Vegetables are dry or frozen.
*   **Root Cause:** EPR (Back-pressure) valve is stuck open or the thermostat is set too low.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Solenoid Click" Test
*   **Trade Trick:** To see if a room is calling for cooling, place a screwdriver on the solenoid valve coil. You should feel a strong magnetic vibration. If the coil is hot but there's no vibration, the internal "Plunger" is stuck. **Hit the valve body gently with a plastic hammer to free it.**

## 5.2 Returning "Trapped Oil"
*   **Trade Trick:** If the compressor oil level is low but there are no leaks, the oil is stuck in the meat room evaporator. **Manually trigger a long defrost cycle** or temporarily raise the meat room temperature to 0°C. The warmth thins the oil, allowing it to flow back to the compressor.

## 5.3 The "Paper Strip" Door Test
*   **Trade Trick:** To check the door seals (gaskets), close the door on a strip of paper. If you can pull the paper out easily, the seal is leaking. **Leaking seals are the #1 cause of evaporator icing.**

## 5.4 Emergency "Hand-Expansion"
*   **Expert Insight:** If the expansion valve fails, most provision plants have a "Manual Bypass" valve. **Warning:** You must adjust this valve by 1/8th turns and watch the suction gauge. If the suction pipe starts to "Frost" all the way to the compressor, you are over-feeding and will destroy the pistons.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Daily Routine
*   **Temp Log:** Record all room temperatures twice daily. This is a mandatory health requirement.
*   **Drain Check:** Ensure the "Scuppers" in the vegetable room are not blocked with rotting leaves.

## 6.2 Monthly Heater Test
*   **Method:** Manually initiate a defrost cycle and use a **Clamp Meter** to check the current draw of the heaters. If the Amps are zero, the heater element is burnt out.

## 6.3 Condenser Cleaning
*   **Method:** Provision condensers are often air-cooled or use a small SW branch. Clean the dust from the air-cooled fins using compressed air (blowing *against* the normal air flow).

---

# Part 7 – Miscellaneous Knowledge

*   **Ethylene Management:** Rotting apples produce Ethylene gas, which makes other vegetables rot faster. **Always keep fruits and vegetables separated** or ensure good ventilation in the Veg room.
*   **Health Inspection:** Port Health Authorities will check the "Provision Temp Log" during inspections. Any temperature above -15°C in the meat room can result in the entire food stock being condemned.

**End of Document**
