# MARINE BOW THRUSTER – COMPLETE CORE KNOWLEDGE

**Equipment:** Transverse Tunnel Thruster (e.g., Kongsberg/Rolls-Royce TT series, Brunvoll FU/LTC series, Schottel STT, Kawasaki KT)  
**Folder Name:** bow_thruster  

**Part 1 – Fundamentals & Physics Behind Operation**  

A transverse tunnel thruster generates lateral thrust by accelerating water through a horizontal tunnel in the bow (or stern). Thrust is produced according to momentum theory:  
\[ T = \dot{m} \Delta v = \rho A v \Delta v \]  
where \( T \) is thrust (kN), \( \dot{m} \) is mass flow rate, \( \Delta v \) is velocity change, \( \rho \) is water density, \( A \) is tunnel cross-sectional area, and \( v \) is water velocity through the propeller.  

The propeller (fixed-pitch or controllable-pitch) imparts axial momentum to the water column inside the tunnel. Direction of thrust is reversed either by motor direction (FPP + VFD) or by changing blade pitch angle (CPP).  

**Speed-loss effect (critical for manoeuvring):** At forward speeds >2 knots a low-pressure zone forms at the tunnel exit and the thruster jet is deflected aft by the hull flow. Thrust efficiency drops ~50 % at 2 knots and becomes negligible above 5 knots (Brunvoll and Schottel installation guidelines).  

**Cavitation physics:** In the confined tunnel, local pressure at blade tips can drop below vapour pressure, causing bubble formation and violent collapse. This produces:  
- Blade-tip erosion (pitting)  
- High-frequency vibration transmitted through the hull  
- Thrust loss (vapour cushion reduces effective mass flow)  

**CPP vs FPP physics:**  
- **CPP (standard on >500 kW units):** Electric motor runs at constant synchronous speed. Thrust magnitude and direction are controlled by hydraulic pitch change (0° neutral → ±25° ahead/astern). Allows instantaneous reversal without high-torque motor starts.  
- **FPP:** Thrust controlled by motor speed and direction via VFD/soft-starter. Simpler mechanically but requires robust electrical cooling for frequent starts/reversals.  

All designs comply with ABS Rules Part 4 Chapter 3 and DNV Pt.4 Ch.5 (thruster independence and power supply requirements).  

**Part 2 – Major Components & Auxiliary Systems**  

**2.1 Prime Mover**  
- Electric motor: 3-phase squirrel-cage induction motor (440 V or HV 6.6/11 kV), high-torque “heavy-start” design.  
- Starting: Soft-starter or VFD to limit in-rush current (≤6 × FLA) and prevent main-switchboard blackout.  
- Hydraulic drive (smaller vessels <300 kW): High-pressure pump + hydraulic motor for excellent torque at zero speed.  

**2.2 Gearbox & Underwater Unit**  
- Right-angle bevel gearbox (spiral bevel gears) converts vertical motor shaft to horizontal propeller shaft.  
- Propeller hub/pod: Houses pitch-change mechanism (hydraulic piston + sliding block for CPP).  
- Shaft seals: Triple-lip or mechanical face seals (most critical component). Seal-oil header tank maintains positive pressure above sea head.  

**2.3 Hydraulic Pitch Control System (CPP)**  
- HPU (Hydraulic Power Unit): Independent electric pump set (20–50 bar, redundant pumps on larger units).  
- Oil Distribution (OD) box: Mounted on input shaft; transfers pressurised oil into rotating shaft via rotary seals.  
- Pitch actuator: Hydraulic cylinder inside hub moves blades via linkage.  
- Feedback: LVDT or potentiometer for exact pitch % to bridge/AMS.  

**2.4 Tunnel & Protective Systems**  
- Tunnel liner: Steel or stainless with epoxy/ceramic coating to resist erosion.  
- Grids/grilles: Prevent debris ingress (reduce thrust ~5–8 % due to drag).  
- Sacrificial zinc anodes on tunnel ends and underwater housing.  
- Resilient mounting (optional on Schottel/Kongsberg high-comfort versions) for noise/vibration reduction.  

**2.5 Auxiliary Systems**  
- Lube-oil system: Gearbox sump + header tank; forced circulation on larger units.  
- Seal-oil system: Separate header tank with low-level and pressure alarms.  
- Control system: PLC-based (integrated with AMS) with interlocks for power availability, pitch-zero start, and low lube-oil pressure.  

**Part 3 – Normal Operation & Key Parameters**  

Healthy operation: Smooth response to pitch commands, minimal vibration, motor current proportional to pitch angle once “loaded”.  

**Typical Healthy Parameters** (Brunvoll/Kongsberg/Schottel data)  

| Parameter                  | Healthy Range                  | Alarm / Trip Threshold          |
|----------------------------|--------------------------------|---------------------------------|
| Seal-oil header tank level | 50–70 % (cold)                 | Low level (leakage)             |
| Seal-oil pressure          | 0.2–0.5 bar > static sea head  | Low pressure (water ingress)    |
| Motor current (zero pitch) | 20–30 % FLA                    | >40 % FLA (bind)                |
| Motor current (full pitch) | 90–95 % FLA                    | >100 % FLA (overload)           |
| Hydraulic system pressure  | 25–45 bar                      | Low <20 bar (no pitch control)  |
| Gearbox lube-oil temp      | 40–60 °C                       | >75 °C (friction)               |
| Vibration                  | <4.5 mm/s RMS                  | >7 mm/s (imbalance)             |

**Seal-oil pressure physics:**  
\[ P_{\text{oil}} > \rho_{\text{water}} \cdot g \cdot h_{\text{draft}} \]  
Ensures any seal leak results in oil outflow (detected by tank level drop) rather than seawater inflow.  

**Part 4 – Common Faults, Symptoms & Root Causes**  

**4.1 Water Ingress (“Milkshake” Fault)**  
Symptoms: Seal-oil tank level rises; oil turns cloudy/emulsified; low seal-oil pressure alarm.  
Root Causes: Rope/fishing line cutting lip seals; degraded O-rings; header-tank pressure loss.  
Real-World Example: Frequent on fishing vessels after net entanglement (Brunvoll service reports).  

**4.2 Pitch Control Hunting / Failure**  
Symptoms: Pitch fluctuates ±5 % without command; slow response.  
Root Causes: Air in hydraulic lines; faulty proportional solenoid valve; worn LVDT/potentiometer.  

**4.3 Motor Trip on Start-Up**  
Symptoms: Instant breaker trip when “Start” pressed.  
Root Causes: Pitch not at zero (propeller loaded); ground fault in motor windings; insufficient available power.  
Interlock: AMS prevents start unless pitch = 0 % and power reserve confirmed.  

**4.4 Excessive Vibration / Noise**  
Symptoms: Heavy thumping or high-pitched grinding.  
Root Causes: Blade damage from debris; gear-tooth spalling; cavitation erosion; loose tunnel grids.  
Detection: Oil analysis shows high Fe (iron) or Cu/Sn (bronze bearings).  

**Part 5 – Expert Troubleshooting Guide & Trade Tricks**  

**Safety Note:** Never override pitch-zero or low-oil interlocks while running. Lock-out/tag-out motor before any work in tunnel.  

**5.1 Emergency Pitch Centering (Minute Trick)**  
If electronic control fails:  
1. Use HPU manual override solenoid valve.  
2. Jog pitch while watching motor amps. Lowest amp reading = zero-pitch (neutral thrust).  

**5.2 Detecting Entanglement Without Diver**  
Run thruster at 10 % pitch both directions.  
- Higher amps or cyclic “tug” in one direction = rope wrapped on shaft.  

**5.3 Header-Tank “Sniff Test”**  
Open cap:  
- Burnt-toast/sulphur smell = gearbox overheating.  
- Frothy oil = air ingress on HPU suction.  

**5.4 Temporary Seal-Oil Pressure Boost (At Sea)**  
If minor leak suspected: Pressurise header tank with 0.2 bar air (or raise tank height if possible) to stop seawater ingress until dry-dock.  

**5.5 Quick Motor-Current Trending**  
Record amps vs pitch % at constant RPM. Any deviation from baseline curve indicates mechanical bind or blade damage.  

**Part 6 – Maintenance & Preventive Checks**  

Refer to OEM manuals for detailed routine schedules and procedures (Kongsberg/Rolls-Royce TT, Brunvoll FU/LTC, Schottel STT project guides and service literature). All intervals, torque values, and acceptance criteria must be taken directly from the vessel-specific approved OEM documentation.  

**Recommended Official PDFs to Download**  
1. Brunvoll FU/LTC Series Instruction Manual (complete with data sheets, maintenance & hydraulics) – https://khinzawshwecom.files.wordpress.com/2018/05/2d-bow-truster-brunvoll-thruster-type-fu-80-ltc-2250-142.pdf  
2. Schottel STT Transverse Thruster Installation & Operation Guide – https://www.schottel.de/fileadmin/downloads/segments/SCHOTTEL_STT.pdf (product brochure with technical data)  
3. Kongsberg / Rolls-Royce Tunnel Thruster Product Documentation & Installation Guidelines – https://www.kongsberg.com/contentassets/65302d65e4b04e129e23b6a233f89168/37.azimuth-2p-22.04.21.pdf (related azimuth/tunnel family)  
4. ABS Rules for Building and Classing Marine Vessels – Part 4 Chapter 3: Propulsion and Steering (thruster requirements) – https://ww2.eagle.org/content/dam/eagle/rules-and-guides/archives/other/1000_marinevessels_2018/mvr-part-4-aug-18.pdf  
5. DNV Rules Pt.4 Ch.5 – Rotating Machinery – Driven Units (tunnel thruster section) – https://www.dnv.com/rules-standards/ (search “transverse thruster”)  
6. Brunvoll Complete Thruster Systems Catalogue – https://pdf.nauticexpo.com/pdf/brunvoll/complete-thruster-systems/194565-107433.html  

**File-Saving Instructions**  
Save this complete document as `00_bow_thruster_complete.md` inside `core_knowledge/bow_thruster/`.  
All content is clean Markdown, ready for Docling ingestion. Place the recommended PDFs in the same folder for offline reference.  

**End of Document**