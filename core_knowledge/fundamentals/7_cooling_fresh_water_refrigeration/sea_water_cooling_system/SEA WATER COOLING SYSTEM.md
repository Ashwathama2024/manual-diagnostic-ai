# MARINE SEA WATER COOLING SYSTEM – COMPLETE CORE KNOWLEDGE

**Equipment:** Main and Auxiliary Seawater Cooling Circuit (Pumps, Sea Chests, and Filters)

**Folder Name:** sea_water_cooling_system

**Prepared by:** Senior Marine Engineer & Fluid Systems Specialist (30+ years field experience)

**Date:** April 2026

---

# Part 1 – Fundamentals & Physics of Seawater Cooling

## 1.1 The Physics of the "Heat Sink"
Seawater is the ultimate heat sink for a ship.
*   **The Physics:** The system must move a mass of water $(\dot{m})$ sufficient to absorb the total waste heat $(Q)$ from the engine, generators, and AC plants while maintaining a reasonable temperature rise ($\Delta T$).
*   **Capacity:** For a large ship, the SW pumps must move thousands of cubic meters per hour. The "Delta T" is typically kept below **10°C** to prevent localized boiling or scale formation in the heat exchangers.

## 1.2 Total Dynamic Head (TDH) Physics
The SW pumps must overcome several physical resistances:
1.  **Static Head:** The height difference between the waterline and the highest cooler.
2.  **Frictional Head:** Resistance from internal pipe surfaces.
3.  **Pressure Drop:** Resistance from the heat exchangers (PHEs) and the MGPS anodes.
*   **The Physics:** $TDH = H_{static} + H_{friction} + H_{equipment}$. If the sea chest strainers are blocked, the $H_{friction}$ at the suction increases, leading to **Cavitation**.

## 1.3 Galvanic and Bio-fouling Physics
*   **Galvanic Corrosion:** Seawater is a strong electrolyte. When different metals (e.g., steel pipe and titanium PHE plates) are connected, a "Battery Effect" occurs, eating away the less noble metal (steel).
*   **Bio-fouling:** Barnacles, mussels, and algae thrive in warm seawater pipes. **The Physics:** Fouling increases the pipe surface roughness, which increases $H_{friction}$ and reduces the effective pipe diameter, leading to a catastrophic loss of cooling capacity.

---

# Part 2 – Major Components & System Layout

## 2.1 Sea Chests (High and Low)
*   **Low Sea Chest:** Located in the bottom shell. Used in deep water.
*   **High Sea Chest:** Located in the side shell. Used in shallow water/ports to avoid sucking in mud and sand.
*   **Physics of Venting:** Each sea chest has a "Vent Pipe" to allow trapped air to escape, ensuring the pumps stay primed.

## 2.2 Main Seawater Pumps
Usually vertical centrifugal pumps.
*   **Material:** Impellers are made of **Nickel-Aluminum Bronze (NiAlBr)** to resist the combined effects of corrosion and high-velocity erosion.

## 2.3 Marine Growth Protection System (MGPS)
*   **Physics:** Uses **Copper (Cu)** and **Aluminum (Al)** or **Iron (Fe)** anodes located in the sea chest.
*   **The Process:** An electrical current dissolves the copper into the water ($Cu^{2+}$ ions). These ions are toxic to mussel larvae, preventing them from settling on the pipe walls. The Aluminum/Iron ions create a protective coating on the internal pipe surfaces to reduce corrosion.

## 2.4 Auto-Backwash Strainers
High-capacity filters (typically 1–3 mm mesh) that automatically clean themselves when the $\Delta P$ rises, ensuring the downstream heat exchangers don't get blocked by shells or seaweed.

---

# Part 3 – Normal Operation & Key Parameters

## 3.1 What "Healthy" Looks Like
*   **Pressure:** Discharge pressure is stable (e.g., 2.5 bar).
*   **Temperature:** Sea water outlet temperature from the Central Coolers is within 5°C – 8°C of the inlet temperature.
*   **MGPS:** Current (Amps) is steady at the control panel, indicating the anodes are dissolving.

## 3.2 Typical Operating Parameters

| Parameter | Healthy Value | Alarm / Limit |
| :--- | :--- | :--- |
| **SW Inlet Temp** | 2°C – 32°C | Climate Dependent |
| **SW Discharge Press** | 2.0 – 3.5 bar | < 1.2 bar (Pump Trip) |
| **PHE ΔP (Seawater)** | 0.3 – 0.6 bar | > 1.0 bar (Fouled) |
| **MGPS Current** | 0.5 – 2.0 Amps | 0.0 Amps (Anode Fail) |

---

# Part 4 – Common Faults & Root Causes

## 4.1 Bio-fouling (Loss of Cooling)
*   **Symptom:** Central cooling temperatures (FW) rise slowly over weeks; SW pump Amps increase.
*   **Root Cause:** MGPS system failed or switched off.
*   **Physics:** Mussels have grown inside the main SW header, restricting flow.

## 4.2 Air-Insuction (Aerate)
*   **Symptom:** SW pressure gauge "jiggles" or fluctuates wildly; loud rattling in the pumps.
*   **Root Cause:** Ship is in heavy weather, and the sea chest is "drawing air" as the hull rolls.
*   **Expert Insight:** Switch to the **Low Sea Chest** and reduce pump speed if possible to maintain prime.

## 4.3 Pinhole Leaks (Erosion-Corrosion)
*   **Symptom:** Saltwater spraying from pipe elbows or near valves.
*   **Root Cause:** High water velocity (> 3 m/s) stripping the protective oxide layer from the steel pipes.
*   **Physics:** Turbulence at elbows causes localized "scouring," exposing fresh metal to the corrosive seawater.

---

# Part 5 – Expert Troubleshooting & Trade Tricks

## 5.1 The "Strainer Vacuum" Trick
*   **Trade Trick:** If you suspect a blockage in the sea chest (plastic bag over the grid), stop the SW pump and **momentarily open the "Working Air" blow-back valve** to the sea chest. The burst of air will push the plastic bag away from the hull.

## 5.2 Detecting a Failing MGPS Anode
*   **Trade Trick:** If the MGPS control panel shows maximum voltage ($V$) but zero current ($A$), the anode has completely dissolved or the cable has snapped. **Expert Insight:** If the voltage is low and current is low, the anode is "Passivated" (coated in slime) and needs a manual cleaning.

## 5.3 The "Dead-End" Temperature Check
*   **Trade Trick:** Use an IR thermometer to check the temperature of "Dead-end" SW branches (like the emergency fire pump suction). If they are hot, you have a leaking isolation valve that is allowing hot cooling water to recirculate.

## 5.4 Starting a Pump in "Brown Water"
*   **Expert Insight:** When in a river or silt-heavy port, **do not run both SW pumps**. The high velocity will suck in massive amounts of mud. Run one pump at the minimum required speed and clean the strainers every hour.

---

# Part 6 – Maintenance & Preventive Checks

## 6.1 Weekly Routine
*   **Strainer Cleaning:** Manually clean the "Standby" sea strainer so it is ready for immediate changeover.
*   **MGPS Check:** Record the anode current and voltage.

## 6.2 Quarterly Anode Inspection
*   **Check:** Diver inspection or borescope check of the MGPS anodes in the sea chest. Replace if they are > 80% consumed.

## 6.3 Annual PHE "Back-flush"
*   **Method:** Reverse the SW flow through the central heat exchangers for 30 minutes. This "blows out" any small shells or sand that have bypassed the strainers.

---

# Part 7 – Miscellaneous Knowledge

*   **Sacrificial Anodes (Zincs):** Standard zinc anodes are used in pump casings and water boxes. They must be replaced every drydock. **Warning:** If you use "Aluminum" anodes in a "Copper" system, you will accelerate the corrosion of the copper.
*   **Recirculation Line:** In Arctic conditions, some of the warm SW discharge is diverted back to the sea chest suction to melt ice and keep the sea chest temperature above freezing.

**End of Document**
