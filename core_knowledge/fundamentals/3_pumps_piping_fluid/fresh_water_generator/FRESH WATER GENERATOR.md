# MARINE FRESH WATER GENERATOR (FWG) – COMPLETE CORE KNOWLEDGE

**Equipment:** Vacuum Evaporator (e.g., Alfa Laval JWP Series, Sasakura, Nirex)

**Folder Name:** fresh_water_generator

**Prepared by:** Senior Marine Engineer & Thermodynamic Systems Expert (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Vacuum Evaporation

## 1.1 The Physics of Boiling Points at Low Pressure
The FWG uses the "Waste Heat" from the Main Engine's jacket cooling water (HT water) to produce fresh water.
*   **The Problem:** HT water is only at 80°C, which is not hot enough to boil seawater at atmospheric pressure (100°C).
*   **The Physics Solution:** By creating a vacuum inside the FWG (approx. **90% vacuum or 0.1 bar absolute pressure**), the boiling point of seawater is lowered to approximately **45°C – 50°C**.

## 1.2 Latent Heat of Evaporation
To turn 1 kg of water at 45°C into steam at 45°C, a massive amount of energy (Latent Heat) must be added.
*   **The Physics:** This energy is provided by the HT jacket water. As the jacket water gives up its heat to the seawater, it is cooled (helping the engine) while the seawater evaporates.

## 1.3 Condensation Physics
Once the steam is formed, it must be turned back into liquid fresh water.
*   **The Physics:** This is done in the "Condenser" section using cold Seawater (SW). The SW absorbs the latent heat from the steam, causing it to condense into droplets.

---

# Part 2 – Major Components & System Layout

## 2.1 The Evaporator (Lower Section)
A plate or tube heat exchanger where hot HT water circulates on one side and seawater on the other.

## 2.2 The Condenser (Upper Section)
A plate or tube heat exchanger where cold SW circulates on one side and the evaporated steam on the other.

## 2.3 The Ejector (Air and Brine)
The "Heart" of the vacuum system.
*   **Physics:** Uses the **Venturi Effect**. High-pressure seawater is pumped through a nozzle, creating a high-velocity jet that "sucks" air and excess seawater (brine) out of the FWG, maintaining the vacuum.

## 2.4 The Demister
A wire mesh located between the evaporator and condenser.
*   **Function:** It catches tiny salt-water droplets (carry-over) that are entrained in the steam, ensuring that only pure water vapor reaches the condenser.

## 2.5 Distillate Pump and Salinometer
*   **Distillate Pump:** Sucks the fresh water from the condenser and pumps it to the storage tanks.
*   **Salinometer:** Measures the electrical conductivity of the water. If the salt content is > 2–10 PPM, it triggers a 3-way valve to dump the water back into the bilge.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Vacuum:** Stable at 90% – 93%.
*   **Production:** Constant flow of clear water (e.g., 20–30 tons/day).
*   **Salinity:** Constant at < 2 PPM.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **Vacuum Level** | 90% – 93% | < 85% (Poor Production) |
| **HT Water Inlet Temp** | 80°C – 85°C | < 70°C (No Production) |
| **SW Feed Flow** | Stable | Low Flow (Scaling Risk) |
| **Salinity** | 0 – 2 PPM | > 10 PPM (Automatic Dump) |
| **Condenser SW Outlet** | 40°C – 45°C | > 50°C (Loss of Vacuum) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Scaling (Hard Water Buildup)
*   **Symptom:** Production slowly drops over several weeks; HT water $\Delta T$ decreases.
*   **Root Cause:** Calcium and Magnesium salts in the seawater "bake" onto the evaporator plates.
*   **Physics:** This happens if the seawater feed rate is too low or the temperature is too high, causing local over-concentration.

## 4.2 High Salinity (Carry-Over)
*   **Symptom:** Salinometer alarm; water being dumped.
*   **Root Cause:** "Priming" or "Carry-over." 
*   **Physics:** If the vacuum is too high or the seawater level is too high, the water "boils" violently, throwing salt-water droplets through the demister.

## 4.3 Loss of Vacuum
*   **Symptom:** Vacuum drops to 50% – 60%; production stops.
*   **Root Cause:** Leaking cover gasket, failed ejector nozzle, or a dry "Air Vent" valve.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Ejector Blockage" Check
*   **Trade Trick:** If the vacuum is poor, check the Ejector SW pressure. If the pressure is high but the vacuum is low, the ejector nozzle is likely blocked by a small shell or piece of plastic.

## 5.2 The "Soap Bubble" Leak Test
*   **Trade Trick:** To find a vacuum leak, slightly pressurize the FWG with 0.2 bar of air and spray soapy water on all gaskets and valves. **Never use high pressure air, or you will blow out the PHE gaskets.**

## 5.3 Optimizing Production with the Bypass
*   **Expert Insight:** If the Main Engine is at low load, you may need to **partially close the HT Bypass valve** to force more hot water through the FWG. Monitor the engine jacket temperature carefully!

## 5.4 Cleaning the Salinometer Electrode
*   **Trade Trick:** If the salinity is high but the water "tastes" fine (expert only!), the salinometer electrode may be oily. Clean it with a soft cloth and a little dish soap.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Chemical Descaling (The "Acid Clean")
*   **Method:** Every 1-3 months (depending on scaling), circulate a weak acid solution (e.g., Sulfamic Acid) through the seawater side of the evaporator. This dissolves the calcium scale without damaging the titanium plates.

## 6.2 Demister Cleaning
*   **Method:** During major overhauls, remove the demister and soak it in fresh water. If it is "caked" with salt, the FWG will never produce high-purity water.

## 6.3 Ejector Pump Maintenance
*   **Method:** The SW ejector pump is the "driver" of the vacuum. Ensure the impeller is not eroded by sand or silt, as a 10% drop in pump pressure can lead to a 50% drop in vacuum efficiency.

---

# Part 7 – Miscellaneous Knowledge

*   **UV Sterilizer / Chlorinator:** The fresh water produced is pure, but it contains no chlorine. It must be passed through a UV sterilizer or a chlorination unit before entering the drinking water tanks.
*   **Mineralizer:** FWG water is "Flat" and can be slightly acidic. A Mineralizer (containing marble chips/limestone) is used to add minerals back into the water for better taste and to protect the ship's piping from corrosion.

**End of Document**
