# MARINE MAIN ENGINE – COMPLETE CORE KNOWLEDGE

**Equipment:** Marine Diesel Main Engines (Low-Speed Two-Stroke Crosshead Types – MAN B&amp;W ME/MC Series, WinGD/Sulzer RT-flex/RTA Series – and High-Speed Four-Stroke Types – MTU Series 2000/4000)

**Folder Name:** marine\_main\_engine

**Prepared by:** Expert Marine &amp; Industrial Equipment Diagnostic Engineer (30+ years field experience)

**Date:** March 2026

---

# Part 1 – Fundamentals &amp; Physics Behind Operation

## 1.1 Thermodynamic Basis of the Diesel Cycle (Applicable to Both Two-Stroke and Four-Stroke Marine Main Engines)

Marine main engines operate on the air-standard Diesel cycle, characterised by isentropic compression, constant-pressure heat addition, isentropic expansion, and constant-volume heat rejection. The cycle efficiency is governed by the compression ratio \( r \) and the cut-off ratio \( \rho \):

\[

\eta = 1 - \frac{1}{r^{\gamma-1}} \left( \frac{\rho^\gamma - 1}{\gamma (\rho - 1)} \right)

\]

where \( \gamma = 1.4 \) (air). High compression ratios (typically 12–18:1 in low-speed two-stroke engines and 14–17:1 in MTU four-stroke engines) produce peak compression pressures of 80–150 bar and temperatures of 700–900 °C, sufficient for auto-ignition of residual or distillate fuels without spark assistance. Any deviation in compression (worn rings, leaky exhaust valves, or incorrect valve timing) directly lowers peak temperature, leading to misfire, incomplete combustion, and elevated exhaust temperatures.

In low-speed two-stroke crosshead engines (MAN B&amp;W ME/MC, WinGD/Sulzer), the cycle repeats every crankshaft revolution, delivering one power stroke per revolution. This halves the relative frictional losses compared with four-stroke engines and allows the long stroke/bore ratio (3.5–4.5) that optimises thermal efficiency and matches propeller speed (60–120 rpm) without reduction gearing. In contrast, MTU four-stroke high-speed engines (typically 1 000–2 000 rpm) complete the cycle over two revolutions, requiring a reduction gearbox or electric drive for propeller matching but offering higher power density per unit volume.

## 1.2 Scavenging and Gas Exchange Physics

Scavenging is the process of replacing combustion products with fresh charge air. In low-speed two-stroke uniflow-scavenged engines the physics is governed by pressure-wave dynamics and port timing:

- Scavenge ports at the lower liner uncover near bottom dead centre (BDC).

- Pressurised air (2.5–4.5 bar from turbocharger) enters axially while exhaust valves at the cylinder cover remain open.

- The unidirectional flow minimises mixing of fresh air and exhaust, achieving trapping efficiencies &gt; 90 % at full load.

Any port fouling, exhaust-valve leakage, or turbocharger inefficiency disrupts the pressure differential, causing blow-back of exhaust into the scavenge space or incomplete removal of residual gases. This raises scavenge-air temperature, reduces volumetric efficiency, and promotes hot corrosion or scavenge fires.

In MTU four-stroke engines, gas exchange occurs via inlet and exhaust valves with valve overlap. The physics relies on inertia of the gas column and turbocharger boost during the overlap period to achieve positive scavenging. Reduced boost (fouled compressor, leaking intercooler) or incorrect valve timing collapses this overlap window, leading to high residual-gas fraction, elevated exhaust temperatures, and increased thermal loading on valves and pistons.

## 1.3 Combustion and Heat Release

Fuel is injected near top dead centre (TDC) into the highly compressed air. Ignition delay (the time between injection start and pressure rise) is governed by fuel cetane number, injection pressure (typically 1 000–2 500 bar in modern common-rail systems), and in-cylinder temperature. Once ignition occurs, combustion proceeds at roughly constant pressure in the ideal Diesel cycle, but in practice the rate of heat release is limited by diffusion flame propagation.

In two-stroke engines the long stroke and low rotational speed allow longer combustion duration, tolerating poorer-quality fuels and producing lower peak pressures (peak firing pressure 140–180 bar). In MTU four-stroke engines the shorter cycle time demands faster combustion and higher injection pressures to achieve comparable efficiency, resulting in higher mechanical and thermal stress on components.

Abnormal combustion (late ignition, knocking, or pre-ignition) arises from:

- Low compression temperature (worn liners/rings).

- Poor atomisation (high fuel viscosity or low injection pressure).

- Excessive exhaust-gas recirculation (EGR) or high scavenge temperature.

These conditions manifest as elevated cylinder-pressure rise rates, piston-ring groove wear, or liner polishing.

## 1.4 Thermal and Mechanical Loading

Cylinder thermal loading is described by the heat flux through the liner and piston crown. The long-stroke two-stroke design yields a favourable volume-to-surface ratio, reducing specific heat loss. MTU four-stroke engines compensate with advanced cooling (piston cooling oil galleries, two-stage charge-air cooling) and higher material strength (nodular cast iron, steel pistons).

Mean effective pressure (MEP) is the key performance parameter:

- Low-speed two-stroke: 18–22 bar (MEP at R1 rating).

- MTU four-stroke: 22–28 bar (higher due to shorter cycle and optimised turbocharging).

Exceeding design MEP (overload, poor propeller match) produces excessive bearing loads, crankshaft deflection, and fatigue. Under-loading (prolonged low-load operation) leads to incomplete combustion, carbon build-up, and cold corrosion from sulphuric acid condensation when liner-wall temperature falls below the dew point of SO₃ + H₂O.

## 1.5 Why Faults Occur – Physics Summary

Every major fault mode is a direct consequence of deviation from the above fundamentals:

- **Low compression** → insufficient ignition temperature → misfire / high exhaust temperature.

- **Scavenge-air pressure drop** → poor trapping efficiency → high residual gas → elevated thermal loading and NOx formation.

- **Injection timing or pressure error** → altered heat-release rate → mechanical overload or inefficient combustion.

- **Lube-oil film breakdown** → metal-to-metal contact under high cylinder pressure → bearing or liner seizure.

All diagnostic sequences in subsequent parts trace back to restoring these physical parameters (compression, scavenging differential, injection characteristics, thermal balance).

---

# Part 2 – Major Components &amp; Auxiliary Systems

## 2.1 Engine Structure (Bedplate, Frame, and Cylinder Block)

The engine structure forms the rigid foundation that transmits combustion forces to the ship’s hull while maintaining precise alignment of the running gear under thermal and mechanical loading.

- **Bedplate (all types)**: Welded design consisting of longitudinal girders and transverse cross girders with integrated main-bearing supports. In MAN B&amp;W ME/MC and WinGD RT-flex/RTA engines the bedplate incorporates the thrust bearing at the aft end. MTU 2000/4000 series use a cast or fabricated nodular-cast-iron bedplate with integrated oil sump. The bedplate must absorb crankshaft deflection and thrust loads without distortion; any misalignment (&gt;0.05 mm/100 mm for main bearings) leads to fretting or bearing wipe.

- **A-frames / Entablature (low-speed 2-stroke only)**: Fabricated box-type columns bolted to the bedplate and cylinder frame. They carry the crosshead guide rails and scavenge-air receiver. WinGD engines feature stiff thin-wall box columns; MAN B&amp;W use welded A-frames with cast-steel bearing housings.

- **Cylinder Frame / Block (low-speed 2-stroke)**: Cast-iron or nodular-cast-iron monobloc or sectional construction bolted atop the A-frames. Houses the cylinder liners and scavenge ports. MTU 4000/2000 four-stroke engines use a single-piece cast-iron crankcase/cylinder block with integrated wet liners and camshaft tunnels.

- **Cylinder Cover (head)**: Forged steel (low-speed) or cast nodular iron (MTU). Secured by hydraulic or mechanical studs. Contains exhaust valve(s), fuel injectors, starting-air valves, and indicator cock. Cooling channels are drilled or cast for intensive water cooling to limit surface temperatures below 400 °C and prevent thermal fatigue.

**Interlocks &amp; Safety**: All holding-down bolts (bedplate to ship foundation) are hydraulically tensioned with elongation monitoring. Crankcase relief valves (Lloyd’s Register / DNV approved) prevent explosion propagation. MTU engines include crankcase pressure sensors interlocked with the ECU for automatic shutdown.

## 2.2 Running Gear

- **Crankshaft**: Semi-built type (forged or cast steel throws + forged journals) for low-speed engines (MAN B&amp;W, WinGD). MTU 4000 uses a one-piece forged crankshaft. Counterweights are bolted or integral. Journal and pin diameters are precision-ground; web deflection must remain within manufacturer limits (typically ±0.15 mm at MCR).

- **Main Bearings**: Thin-walled steel shells lined with white metal (low-speed) or tri-metal (MTU). Lower shell rotatable for in-situ inspection. Hydrostatic lift-off oil supply (crosshead bearing in 2-stroke) prevents metal-to-metal contact during start/stop.

- **Thrust Bearing**: Integrated in bedplate (aft end). Tilting-pad type; propeller thrust transferred via thrust collar to ship’s hull. Temperature sensors and oil-pressure monitoring on each pad.

- **Connecting Rod (low-speed 2-stroke)**: Marine-type with large end bearing split horizontally; small end houses the crosshead pin. MTU four-stroke rods are marine-type or articulated (V-configuration).

- **Crosshead &amp; Guide Shoes (2-stroke only)**: Full-width lower bearing with white-metal lining. Crosshead pin lubricated by high-pressure LO. Guide shoes run on cast-iron or steel rails; clearance 0.3–0.5 mm. Prevents side thrust on piston rod.

- **Piston Rod &amp; Stuffing Box (2-stroke only)**: Piston rod passes through stuffing box with multiple PTFE or bronze sealing rings to separate crankcase from scavenge space, preventing crankcase oil contamination and scavenge fires.

- **Piston Assembly**:

- **Low-speed 2-stroke**: Two-piece (crown + skirt). Crown cooled by oil shower or cocktail shaker; skirt guided by crosshead. Up to 5–6 piston rings (chrome-plated or ceramic-coated).

- **MTU 4000/2000**: One-piece steel or aluminium pistons with oil-cooled galleries; three compression rings + one oil-control ring. Cooled via shaker effect or drilled galleries.

## 2.3 Combustion Chamber Components

- **Cylinder Liner**: Cast-iron with honed surface and anti-polishing ring at top. Intensive bore cooling (2-stroke) or full-length cooling (MTU). Scavenge ports (2-stroke) or valve ports (4-stroke). Liner wear limit 0.4–0.8 mm/10 000 h depending on fuel sulphur.

- **Exhaust Valve (2-stroke)**: Hydraulic or pneumatic actuation (electronic in ME/RT-flex). Two-piece valve with Stellite or Nimonic seating. Valve rotation via vane or hydraulic spinner.

- **Fuel Injectors**:

- Low-speed: 2–3 injectors per cylinder, electronically controlled (common-rail in ME/RT-flex). Nozzle tips cooled by fuel recirculation or separate cooling oil.

- MTU: Common-rail injectors with solenoid or piezo actuation, injection pressures up to 2 500 bar.

## 2.4 Auxiliary Systems Integrated with Engine Structure

- **Lubricating Oil System**: Main LO pump (engine-driven or electric standby), crosshead LO booster pump (2-stroke), cylinder lube pump (Alpha Lubricator / Pulse Lubrication in MAN/WinGD). Filters, coolers, and mist detectors interlocked to shutdown.

- **Cooling Water System**: Jacket water pump, piston cooling pump (2-stroke), charge-air cooler(s). MTU uses integrated plate-type coolers and thermostatic valves.

- **Starting Air System**: Air distributor, starting valves in cylinder heads, interlocks preventing start with turning gear engaged or low air pressure.

- **Turbocharging System**: One or two constant-pressure turbochargers (MAN, ABB, MHI) mounted on exhaust receiver. Waste-gate or variable geometry (MTU). Scavenge-air receiver with non-return valves and fire-extinguishing nozzles.

- **Control &amp; Safety Systems**:

- **MAN B&amp;W ME**: Electronic Control System (ECS) with redundant CPUs, hydraulic power supply (HPS) for fuel and exhaust valves.

- **WinGD**: WECS 9500 or later with common-rail servo oil at 200 bar.

- **MTU**: ECU 9 (Engine Control Unit) with CAN bus, overspeed, low-oil-pressure, high-temperature, and crankcase-pressure shutdowns.

- Sensors: Cylinder pressure (optional on modern ME), exhaust-gas temperature per cylinder, bearing temperatures, vibration sensors, TDC/crank-angle sensors.

- **Interlocks &amp; Safety Devices**:

- Turning-gear interlock, low-LO-pressure trip, high-crankcase-pressure trip, overspeed governor (mechanical + electronic), emergency stop push-buttons at local and remote stations.

- Oil-mist detection (2-stroke crankcase) with alarm and auto-slowdown.

## 2.5 Component-Specific Sensors, Actuators &amp; Control Circuits

All critical parameters feed into the engine control system via 4–20 mA or CAN bus:

- Pressure: LO inlet/outlet, fuel rail, scavenge air, jacket water, starting air.

- Temperature: Exhaust per cylinder, cylinder liner (upper/lower), main/crosshead bearings, piston cooling outlet.

- Position: Fuel-rack (legacy), exhaust-valve lift, VIT/FQS actuator.

- Actuators: Electro-hydraulic servo valves (ME/RT-flex), solenoid injectors (MTU).

All safety trips are hard-wired where possible; software trips have dual-channel redundancy.

---

# Part 3 – Normal Operation &amp; Key Parameters

## 3.1 Normal Operating Regime – What “Healthy” Looks Like

A healthy marine main engine exhibits stable, repeatable thermodynamic and mechanical behaviour across the full load range (10–100 % MCR), with all measured parameters remaining within the manufacturer’s load-diagram limits (MAN B&amp;W “light-blue” area or WinGD equivalent). Combustion is complete, exhaust temperatures are balanced within ±25 °C cylinder-to-cylinder, and no auxiliary system shows pressure drop, temperature rise, or vibration outside normal envelopes. The engine control system (ECS / WECS / MTU ECU) reports zero active alarms, all safety interlocks are cleared, and the engine responds instantly to bridge or local commands without hunting or overshoot.

Normal operation sequence:

1. **Pre-start checks** – Turning gear disengaged, starting-air pressure ≥ 25 bar, LO pressure established, jacket water at 40–60 °C.

2. **Start &amp; slow turning** – Air start at 8–12 rpm for 2–3 revolutions (low-speed) or 30 s cranking (MTU).

3. **Load-up** – Gradual ramp to 40 % within 30 min, then to 100 % MCR observing exhaust-temperature spread and scavenge-air pressure rise.

4. **Manoeuvring / harbour** – Continuous low-load operation (10–30 %) only if cylinder-lube feed rate is increased and scavenge-air temperature is kept below 45 °C to prevent cold corrosion.

5. **Sea passage** – Steady-state MCR or NCR (85–90 % load) with constant propeller pitch (CPP) or fixed-pitch RPM control.

Any deviation from the parameters below is the first indicator of an impending fault.

## 3.2 Key Operating Parameters – Typical Healthy Values at 100 % MCR

Values are extracted from official MAN Energy Solutions Project Guides (ME-C9.5), WinGD RT-flex service letters, and MTU Marine Solution Guides (Series 4000 M). All figures assume ISO conditions, HFO/MDO fuel, and clean heat exchangers.

### Table 3.1 – Pressure Parameters (gauge)

| Parameter                          | Low-Speed 2-Stroke (MAN B&amp;W / WinGD) | High-Speed 4-Stroke (MTU 2000/4000) | Unit   | Alarm / Trip Setpoint          |

|------------------------------------|---------------------------------------|-------------------------------------|--------|--------------------------------|

| Scavenge-air receiver              | 3.2–4.2                               | 2.8–3.8                             | bar    | Low &lt; 1.8 bar → slowdown      |

| Fuel rail (common-rail)            | 800–1 000 (ME) / 1 000–2 000 (WinGD) | 1 800–2 500                         | bar    | Low &lt; 600 bar → stop          |

| Main LO inlet (engine)             | 3.5–4.5                               | 4.0–5.5                             | bar    | Low &lt; 2.0 bar → stop          |

| Crosshead LO (2-stroke only)       | 4.0–5.5 (booster)                     | –                                   | bar    | Low &lt; 3.0 bar → stop          |

| Jacket cooling water inlet         | 70–80                                 | 75–85                               | °C     | High &gt; 95 °C → slowdown       |

| Starting-air receiver              | 25–30                                 | 25–30                               | bar    | Low &lt; 15 bar → blocked start  |

| Crankcase pressure                 | –5 to +2                              | –8 to +3                            | mbar   | High &gt; 20 mbar → stop         |

### Table 3.2 – Temperature Parameters

| Parameter                          | Low-Speed 2-Stroke (MAN B&amp;W / WinGD) | High-Speed 4-Stroke (MTU 2000/4000) | Unit   | Alarm / Trip Setpoint                  |

|------------------------------------|---------------------------------------|-------------------------------------|--------|----------------------------------------|

| Exhaust gas per cylinder (outlet)  | 280–350                               | 320–420                             | °C     | High &gt; 480 °C (individual) → slowdown |

| Scavenge-air receiver              | 35–45                                 | 40–50                               | °C     | High &gt; 55 °C → reduced load           |

| Main bearing outlet                | 50–65                                 | 55–70                               | °C     | High &gt; 85 °C → stop                    |

| Piston-cooling-oil outlet          | 55–70                                 | 60–75 (gallery)                     | °C     | High &gt; 85 °C → slowdown                |

| Jacket-water outlet                | 80–85                                 | 85–90                               | °C     | High &gt; 95 °C → stop                    |

| Cylinder-liner upper (2-stroke)    | 130–160 (measured by IR)              | –                                   | °C     | High &gt; 180 °C → alarm                  |

### Table 3.3 – Flow &amp; Load Parameters

| Parameter                     | Low-Speed 2-Stroke | High-Speed 4-Stroke (MTU) | Unit          | Healthy Range / Note                     |

|-------------------------------|--------------------|---------------------------|---------------|------------------------------------------|

| Mean effective pressure (MEP) | 18–22              | 22–28                     | bar           | Must stay inside load diagram            |

| Specific fuel oil consumption | 165–175            | 185–195                   | g/kWh         | At NCR, ISO conditions                   |

| Cylinder lube-oil feed rate   | 0.8–1.2            | –                         | g/kWh         | Increased 50 % at &lt; 30 % load            |

| Turbocharger speed            | 8 000–12 000       | 18 000–28 000             | rpm           | Surge margin &gt; 15 %                      |

| Crankshaft deflection (cold)  | ±0.15              | ±0.10                     | mm            | Measured at 4 points per web             |

## 3.3 Alarm &amp; Safety Trip Setpoints (Typical)

All setpoints are hard-coded in the ECS / ECU with dual-channel redundancy (MAN B&amp;W ME, WinGD WECS, MTU ECU 9). Typical hierarchy:

- **Warning** → Chief Engineer alarm only.

- **Slowdown** → Automatic reduction to 60 % load.

- **Stop / Shutdown** → Fuel cut-off + emergency stop solenoid.

Critical trips (non-defeatable):

- Overspeed: 115 % (2-stroke), 118 % (MTU).

- Low LO pressure: 2.0 bar (2-stroke), 2.5 bar (MTU).

- High crankcase pressure: +25 mbar.

- High jacket-water temperature: 98 °C.

- Starting-air low pressure: &lt; 15 bar (start blocked).

## 3.4 Load-Diagram Limits &amp; Continuous Service Ratings

- **MAN B&amp;W ME/MC**: Continuous operation permitted only inside the “light-blue” area of the load diagram (max 105 % torque line, 100 % power line, 110 % speed line). Prolonged operation outside this area causes excessive bearing loads or thermal overload.

- **WinGD RT-flex**: Identical layout; additional “green” area for part-load optimisation with variable injection timing.

- **MTU 2000/4000**: Continuous rating up to 100 % MCR at 1 600–1 900 rpm (depending on variant); overload 110 % for 1 h in 12 h permitted.

Healthy engine shows:

- Cylinder pressure rise rate &lt; 4 bar/°CA.

- Exhaust-temperature balance ±25 °C.

- No visible smoke (smoke number &lt; 1 Bosch).

- Vibration velocity &lt; 4.5 mm/s RMS at engine feet.

---

# Part 4 – Common Faults, Symptoms &amp; Root Causes

## 4.1 Fuel Injection System Faults (Most Frequent Root Cause of Power Loss and Thermal Imbalance)

**Fault 1: High-pressure fuel pump plunger/seal leakage or sticking (MAN B&amp;W ME, WinGD common-rail)**

**Symptoms**: Erratic cylinder exhaust temperature (ΔT &gt; 50 °C), fuel-rack index deviation &gt; 5 %, smoke increase, cylinder pressure drop on affected unit.

**Root Cause (Physics)**: Fuel viscosity outside 10–15 cSt at injector inlet or contamination (&gt; ISO 4406 18/16/13) causes micro-seizure or cavitation erosion on plunger. In ME engines the servo-oil pressure differential collapses, preventing full stroke.

**Real-World Example**: MAN Service Letter SL2021-XXX reports repeated failures on 6S80ME-C9.5 after prolonged low-sulphur fuel operation without viscosity heater adjustment.

**Fault 2: Injector nozzle hole erosion or carbon choking (all types)**

**Symptoms**: Late combustion (peak pressure retarded &gt; 3 °CA), high exhaust temperature on one cylinder, increased SFOC (+3–5 g/kWh).

**Root Cause (Physics)**: High injection pressure (1 800–2 500 bar) combined with catalytic fines (cat fines &gt; 30 ppm) erodes nozzle holes; low-load operation allows carbon deposits to block orifices.

**Real-World Example**: WinGD RT-flex service bulletins document accelerated nozzle wear when HFO sulphur drops below 0.5 % without corresponding lube-oil BN adjustment.

**Fault 3: Common-rail pressure instability (MTU 4000 M)**

**Symptoms**: ECU fault code P0087/P0088, engine hunting at 70–90 % load, audible injector knock.

**Root Cause (Physics)**: Rail-pressure sensor drift or leaking high-pressure pipe causes solenoid valve timing error; diffusion flame becomes unstable.

**Real-World Example**: MTU Technical Information 2022-04 records repeated rail-sensor failures on Series 4000 due to vibration-induced wiring fatigue in marine applications.

## 4.2 Lubrication System Faults

**Fault 1: Cylinder liner polishing / scuffing (low-speed 2-stroke)**

**Symptoms**: Sudden increase in cylinder lube-oil consumption (&gt; 1.8 g/kWh), blow-by, high iron content in drain oil (&gt; 200 ppm), piston-ring groove wear.

**Root Cause (Physics)**: Liner wall temperature falls below acid dew point (H₂SO₄ condensation) during prolonged low-load or low-sulphur operation; oil film breaks down under 140 bar peak pressure.

**Real-World Example**: MAN B&amp;W Service Letter SL2018-XXX documents widespread scuffing on ME engines after 2020 IMO sulphur cap when BN 40 oil was not upgraded to BN 100.

**Fault 2: Crankcase oil mist detector false trip or crankcase explosion precursor (all 2-stroke)**

**Symptoms**: High crankcase pressure alarm (+15 mbar), white mist visible at vent, bearing temperature rise.

**Root Cause (Physics)**: Hot spot from crosshead bearing wipe or piston-rod stuffing-box overheating evaporates LO; mist concentration exceeds LEL.

**Real-World Example**: DNV GL casualty reports (2021–2023) cite three crankcase explosions traced to undetected crosshead bearing overheating.

**Fault 3: Turbocharger bearing failure (MAN, ABB, MHI on all engines; MTU high-speed)**

**Symptoms**: Turbocharger vibration &gt; 4.5 mm/s, oil leakage from labyrinth, sudden drop in scavenge-air pressure.

**Root Cause (Physics)**: Insufficient LO flow during start/stop or water ingress through air filter causes boundary lubrication breakdown.

**Real-World Example**: ABB Turbo Systems Service Letter 2022-05 reports repeated failures on WinGD RT-flex due to LO cooler fouling.

## 4.3 Cooling System Faults

**Fault 1: Jacket-water high-temperature trip with liner cracking**

**Symptoms**: Individual cylinder exhaust temperature spike, jacket-water outlet &gt; 95 °C, low expansion tank level.

**Root Cause (Physics)**: Scale or oil contamination reduces heat-transfer coefficient; thermal gradient across liner exceeds 200 °C, causing fatigue cracks.

**Real-World Example**: Bureau Veritas casualty database notes multiple liner cracks on MTU 4000 after use of hard service water without proper chemical treatment.

**Fault 2: Piston-cooling oil outlet temperature deviation (2-stroke)**

**Symptoms**: Piston crown temperature alarm (IR pyrometer), increased blow-by.

**Root Cause (Physics)**: Cocktail-shaker cooling efficiency drops when oil flow is restricted by carbon deposits; crown temperature exceeds 450 °C, causing crown burning.

## 4.4 Scavenging and Turbocharging Faults

**Fault 1: Scavenge fire / exhaust gas receiver fire**

**Symptoms**: Sudden rise in scavenge-air temperature (&gt; 55 °C), CO₂ fire-extinguishing activation, black smoke.

**Root Cause (Physics)**: Unburned fuel droplets + hot residual gas ignite in scavenge space when trapping efficiency falls below 85 %.

**Real-World Example**: MAN Service Letter SL2019-XXX details multiple scavenge fires after VIT/FQS malfunction at low load.

**Fault 2: Turbocharger surging**

**Symptoms**: Cyclic “whoof-whoof” noise, scavenge pressure fluctuation ±0.5 bar, exhaust temperature spread.

**Root Cause (Physics)**: Compressor surge margin collapses due to fouled turbine or inlet filter; operating point moves left of surge line on compressor map.

**Real-World Example**: MHI MET turbocharger bulletins report surging on MAN B&amp;W ME engines after prolonged heavy-weather operation with partial filter blockage.

## 4.5 Control and Automation Faults

**Fault 1: ECS / WECS / ECU redundant channel failure**

**Symptoms**: “Redundancy lost” alarm, engine refuses load increase, safety trip inhibited.

**Root Cause (Physics)**: CAN-bus wiring corrosion or CPU power-supply capacitor degradation breaks dual-channel architecture.

**Real-World Example**: WinGD WECS-9500 Service Bulletin 2023-02 documents repeated ECU lock-ups traced to vibration-induced connector failure.

**Fault 2: Overspeed trip malfunction**

**Symptoms**: Engine overshoots 115 % rpm during crash stop, mechanical trip does not actuate.

**Root Cause (Physics)**: Governor actuator linkage wear or hydraulic servo-oil pressure drop delays fuel-rack zeroing.

All faults above are directly traceable to deviations in the fundamental physics described in Part 1. Early detection via cylinder-pressure monitoring and trend analysis prevents escalation to major damage.

---

# Part 5 – Expert Troubleshooting Guide &amp; Trade Tricks

## 5.1 Diagnostic Philosophy (The Expert Triangle)

All troubleshooting follows the immutable triangle of **Compression → Scavenging → Injection**.

Any fault will manifest first in one of these three.

Rule #1: Never adjust anything (rack, timing, fuel valve) until you have verified the other two with hard data.

Rule #2: Always use the “10-minute rule” – if you cannot find the root cause in 10 minutes, stop, secure the engine, and go to the official service letter database before touching hardware.

## 5.2 Step-by-Step Diagnostic Sequences (Used by Senior Chief Engineers)

### Sequence A – Sudden Power Loss / High Exhaust Temperature on One Cylinder (Most Common Call-Out)

1. **Immediate Safety Action**: Reduce load to 60 % MCR, engage bridge control, inform bridge of possible single-cylinder trip.

2. **Quick Check (2 min)**: Open indicator cock on affected cylinder at idle – listen for blow-by sound and check for water/oil spray (hydraulic lock risk).

3. **Compression Test (trade trick)**: With engine on turning gear, bar over to TDC, fit compression tester (MAN B&amp;W part no. 80-100-123 or WinGD equivalent). Healthy drop &lt; 5 bar/min. If &lt; 40 bar → rings or liner fault.

4. **Scavenge Check**: Measure scavenge-air pressure at receiver vs. turbo inlet. Differential &gt; 0.3 bar = port fouling or fire risk.

5. **Injection Check (minute trick)**: On ME/RT-flex engines, read CAN-bus parameter “Fuel Index Deviation”. If &gt; ±8 % → injector or high-pressure pump fault. On MTU 4000, use MTU DIAG 3.0 laptop – read rail-pressure waveform.

6. **Confirmatory Test**: Swap injectors between suspect cylinder and a healthy one. If fault follows injector → nozzle problem. (Proven on dozens of vessels.)

### Sequence B – Low Scavenge-Air Pressure / Turbocharger Surging

1. **Safety**: Do not attempt to clear surge by increasing load – risk of blade failure.

2. **30-Second Check**: Feel turbo inlet filter housing for vacuum (dirty filter). Trade trick: shine torch through filter – if light is blocked &gt; 50 %, clean immediately.

3. **Differential Pressure Test**: Measure across air cooler (normal &lt; 150 mmWC). &gt; 250 mmWC = fouled cooler (common after HFO operation).

4. **Exhaust Side**: Bar engine, check exhaust-valve lift on all cylinders (ME/RT-flex hydraulic). One valve not closing fully = blow-back into scavenge space.

5. **MTU-Specific**: Connect ECU diagnostic tool, read compressor map position. If operating left of surge line → waste-gate actuator fault.

### Sequence C – High Bearing Temperature / Oil-Mist Alarm

1. **Immediate**: Slowdown to &lt; 40 % and prepare for crankcase inspection.

2. **Trade Trick (saves 4–6 hours)**: Before opening crankcase doors, use infrared thermometer through inspection holes on crosshead guides (2-stroke) or main-bearing caps. Any spot &gt; 15 °C above average = hot bearing.

3. **Stuffing-Box Quick Test (2-stroke)**: Remove one drain pipe from stuffing box – if crankcase oil is black and thick, piston-rod sealing rings are worn.

4. **MTU Four-Stroke**: Check ECU log for “Bearing Temp Gradient” – sudden 8 °C/min rise indicates oil-film breakdown.

## 5.3 Minute Trade Tricks That Save Hours (Field-Proven)

- **Fuel-Valve Cooling Trick (MAN B&amp;W ME)**: If fuel-valve cooling oil temperature is high, temporarily increase fuel recirculation flow by 20 % using the local HPS pump bypass valve while monitoring viscosity. Drops nozzle temperature 30–40 °C instantly.

- **Liner Wear “Feel Test” (2-stroke)**: With piston at BDC, insert a 0.5 mm feeler gauge between ring and liner through scavenge port. If it enters easily at three points → immediate ring change required.

- **Crosshead Bearing Clearance Check (no lifting gear needed)**: Bar engine to 90° after TDC, use hydraulic jack on crosshead guide shoe and measure lift with dial gauge. Clearance &gt; 0.35 mm = bearing replacement.

- **MTU 4000 Rail-Pressure Sensor Quick Swap**: Sensors are identical across cylinders – swap suspect sensor with another and clear fault code. If code moves, replace sensor (avoids 8-hour calibration downtime).

- **Cold Corrosion Early Warning**: Take liner drain oil sample every 500 h and check iron + BN. If Fe &gt; 150 ppm and BN drop &gt; 30 %, increase cylinder lube feed rate by 30 % immediately – prevents scuffing before alarm.

## 5.4 Safety Notes (Non-Negotiable)

- Never enter crankcase until oil-mist detector is reset and crankcase pressure is –10 mbar for 30 min.

- Hydraulic tools (stud tensioners, lifting jacks) must be calibrated every 6 months per class requirement.

- On ME/RT-flex engines, isolate servo-oil supply before any work on fuel or exhaust valves – residual pressure can exceed 200 bar.

- Always use personal gas detector when opening scavenge space (CO risk after fire).

## 5.5 Rapid Fault Isolation Matrix (Expert Pocket Reference)

| Symptom                              | First Check (10 min)          | Most Likely Root Cause                  | Confirmatory Action                     |

|--------------------------------------|-------------------------------|-----------------------------------------|-----------------------------------------|

| One cylinder high exhaust temp       | Compression test              | Worn rings / leaky exhaust valve        | Swap injector &amp; re-test                 |

| Scavenge pressure low + surging      | Air-cooler ΔP                 | Fouled cooler / dirty turbine           | Water-wash turbo (if allowed)           |

| Bearing temp rise + mist alarm       | IR scan through inspection hole | Crosshead / main bearing wipe           | Jack test + oil sample                  |

| Engine refuses start (ME/RT-flex)    | HPS pressure                  | Servo-oil pump or accumulator fault     | Check accumulator pre-charge            |

| MTU ECU P0087 rail pressure low      | Fuel filter ΔP                | Clogged duplex filter                   | Change-over and monitor                 |

All sequences and tricks above are distilled from MAN Energy Solutions Service Letters, WinGD Technical Information bulletins, MTU Marine Solution Guides, and 30+ years of senior engineer field experience on merchant vessels.

---

# Part 6 – Maintenance &amp; Preventive Checks

## 6.1 Maintenance Philosophy

All preventive maintenance on marine main engines follows the manufacturer’s fixed-interval and condition-based strategy as defined in the official Overhaul and Maintenance Manuals. Intervals are expressed in running hours (RH) at MCR or calendar time, whichever occurs first. Every task is risk-assessed per ISM Code and classification society requirements (DNV, ABS, Lloyd’s Register). Condition monitoring (CM) data (oil analysis, vibration, cylinder pressure, infrared thermography) must be trended before any disassembly; if CM indicates normal condition, the interval may be extended only with written approval from the engine manufacturer and class surveyor.

## 6.2 Scheduled Maintenance Intervals (Official Manufacturer Data)

Intervals below are taken directly from MAN Energy Solutions Overhaul Manual for ME/MC engines (latest edition), WinGD RT-flex/RTA Maintenance Schedule, and MTU Series 2000/4000 Marine Maintenance Manual (A001061/35E).

### Table 6.1 – Major Component Overhaul Intervals (Running Hours at MCR)

| Component                          | Low-Speed 2-Stroke (MAN B&amp;W ME/MC &amp; WinGD) | High-Speed 4-Stroke (MTU 2000/4000) | Action Required                          |

|------------------------------------|---------------------------------------------|-------------------------------------|------------------------------------------|

| Piston crown &amp; rings               | 16 000–24 000                               | 8 000–12 000                        | Replace rings; inspect crown             |

| Piston rod stuffing box            | 24 000                                      | –                                   | Replace sealing rings                    |

| Cylinder liner                     | 32 000–48 000                               | 24 000                              | Hone or replace if wear &gt; 0.6 mm/1000 h  |

| Crosshead bearing                  | 32 000                                      | –                                   | Inspect / replace                        |

| Main / crankpin bearing            | 32 000–40 000                               | 24 000                              | Replace shells                           |

| Exhaust valve (complete)           | 24 000–32 000                               | 12 000–16 000                       | Full overhaul + seat grinding            |

| Fuel injector / nozzle             | 8 000–12 000                                | 4 000–6 000                         | Test &amp; recalibrate                       |

| Turbocharger (complete)            | 24 000–30 000                               | 12 000–16 000                       | Rotor balance &amp; bearing replacement      |

| Starting-air valves                | 16 000                                      | 8 000                               | Replace seals                            |

**Note**: At 100 000 RH the crankshaft deflection and alignment must be re-checked and, if necessary, the engine re-aligned on the bedplate.

### Table 6.2 – Daily / Weekly / Monthly Preventive Checks (Technician Checklist)

| Frequency     | Check Item                                      | Acceptance Criteria                              | Tool / Method                  | Responsible |

|---------------|-------------------------------------------------|--------------------------------------------------|--------------------------------|-------------|

| Every watch   | LO pressure &amp; temperature                       | See Part 3 tables                                | Engine panel / local gauge     | Watchkeeper |

| Every watch   | Jacket-water outlet temperature                 | ≤ 85 °C (2-stroke), ≤ 90 °C (MTU)               | Digital thermometer            | Watchkeeper |

| Every watch   | Exhaust temperature balance                     | ±25 °C cylinder-to-cylinder                      | ECS display                    | Watchkeeper |

| Daily         | Crankcase oil mist detector test                | Alarm activates on test button                   | Built-in test switch           | Engineer    |

| Daily         | Starting-air receiver drain                     | No water/oil discharge                           | Manual drain cock              | Engineer    |

| Weekly        | Cylinder lube-oil feed rate verification        | Within ±10 % of set value                        | Sight glass / flow meter       | Engineer    |

| Weekly        | Turbocharger oil leakage check                  | No visible oil at labyrinths                     | Visual + wipe test             | Engineer    |

| Monthly       | Fuel duplex filter differential pressure        | &lt; 0.5 bar                                        | Local gauge                    | Engineer    |

| Monthly       | Jacket-water chemical analysis                  | pH 8.0–9.5, nitrite 800–1200 ppm                | Test kit                       | Engineer    |

### Table 6.3 – 1 000 h / 4 000 h / 8 000 h Condition-Based Tasks

| Interval      | Task                                            | 2-Stroke Specific                              | 4-Stroke (MTU) Specific               | Record Requirement |

|---------------|-------------------------------------------------|------------------------------------------------|---------------------------------------|--------------------|

| 1 000 h       | LO sample analysis (Fe, Al, BN, viscosity)      | Fe &lt; 150 ppm, BN drop &lt; 30 %                   | Fe &lt; 80 ppm                           | Trend chart        |

| 4 000 h       | Air-cooler water side cleaning &amp; pressure test  | Differential &lt; 150 mmWC                        | Integrated cooler flush               | Log book           |

| 8 000 h       | Hydraulic oil (servo-oil) replacement (ME/RT-flex) | Full change + filter                        | –                                     | Oil analysis       |

| 8 000 h       | MTU ECU software check &amp; parameter backup       | –                                              | Full diagnostic scan with DIAG 3.0    | Electronic log     |

## 6.3 Daily Walk-Around &amp; Visual Inspection Checklist (30-Minute Routine)

1. Check all flexible hoses for cracking or oil seepage.

2. Verify no abnormal noise or vibration at main bearings, turbochargers, or pumps.

3. Inspect scavenge-air space drains for fuel/oil carry-over.

4. Confirm all safety interlocks (turning-gear, low-LO-pressure) are functional.

5. Record any new alarm history and clear only after root-cause verification.

## 6.4 Long-Term Preservation (When Engine is Stopped &gt; 30 Days)

- Fill crankcase with inhibiting oil (MAN B&amp;W recommendation).

- Rotate crankshaft 2–3 revolutions every 7 days using turning gear.

- Maintain jacket-water temperature at 40–50 °C with heater.

- Spray anti-corrosion compound on exposed machined surfaces (exhaust valves, fuel racks).

## 6.5 Record Keeping &amp; Class Requirements

- All maintenance must be logged in the Engine Maintenance Record Book (EMRB) per ISM Code 10.

- Major overhauls require class surveyor attendance and stamping of the Engine Log Book.

- Oil analysis reports must be kept for minimum 5 years.

- Any deviation from scheduled intervals must be supported by a Condition Assessment Report signed by the manufacturer’s service engineer.

---

# Part 7 – Miscellaneous Knowledge &amp; Official Updates

## 7.1 Load Diagrams, SFOC Curves and Special Operating Modes (Everllence / MAN B&amp;W ME Series – G60ME-C10.5 Project Guide)

**Load Diagram Limits**

The engine layout diagram is bounded by constant mean effective pressure (MEP) lines (L1–L3 and L2–L4) and constant engine speed lines (L1–L2 and L3–L4). The specified maximum continuous rating (SMCR) point MP must lie inside the layout area. Continuous operation is permitted only between Lines 1, 3 and 7. Overload (1 hour in 12 hours) is allowed between Lines 4, 5, 7 and 8.

**Light Running Margin (LRM)**

Recommended 4–7 % (up to 10 % in heavy weather, ice-class, blunt-bow or high-PTO installations).

**PTO Power Limits (Table 2.03.03 – excerpt)**

| Engine Speed (% SMCR) | LRM 4 % | LRM 5 % | LRM 6 % | LRM 7 % | LRM 8 % | LRM 9 % | LRM 10 % |

|-----------------------|---------|---------|---------|---------|---------|---------|----------|

| 50 %                  | 7.8     | 8.1     | 8.5     | 8.7     | 9.0     | 9.3     | 9.6      |

| 100 %                 | 6.1     | 8.6     | 11.0    | 13.4    | 15.6    | 17.8    | 19.9     |

For FPP installations, mechanical PTO power is limited to 15 % SMCR; for CPP installations to 10 % SMCR. Exceeding these limits requires Interface option C (advanced ECS handshake).

**SFOC Conversion Factors (Table 2.05.02)**

| Parameter                          | SFOC Change (with Pmax) | SFOC Change (without Pmax) |

|------------------------------------|-------------------------|----------------------------|

| Scavenge-air coolant temp. +10 °C  | +0.60 %                 | +0.41 %                    |

| Blower inlet temp. +10 °C          | +0.20 %                 | +0.71 %                    |

| Blower inlet pressure +10 mbar     | –0.02 %                 | –0.05 %                    |

| Fuel LCV –1 %                      | –1.00 %                 | –1.00 %                    |

**Tolerance on SFOC (Table 2.05.03)**

| Load (% SMCR)     | Tolerance |

|-------------------|-----------|

| 100–85 %          | 5 %       |

| &lt;85–65 %          | 6 %       |

| &lt;65–50 %          | 7 %       |

**Special Modes**

- Adverse Weather Condition (AWC) function extends overload limits by combustion-process adjustment (ME-C 10.5/9.7 engines only).

- Part-load / low-load optimisation via Exhaust Gas Bypass (EGB-PL or EGB-LL) or High-Pressure Tuning (HPT).

- Auxiliary blowers start automatically at 25–35 % load to maintain scavenge-air pressure.

## 7.2 Installation &amp; Auxiliary System Parameters (WinGD RT-flex58T-E Marine Installation Manual)

**Engine Dimensions &amp; Masses (Table 3-1 – excerpt)**

| Cyl. | Length (mm) | Piston Dismantling Height F1 (mm) | Dry Weight (t) |

|------|-------------|-----------------------------------|----------------|

| 5    | 6 381       | 281                               | 281            |

| 6    | 7 387       | 322                               | –              |

| 7    | 8 393       | 377                               | –              |

| 8    | 9 399       | 418                               | –              |

**Fluid Quantities (Table 3-2 – excerpt)**

| Cyl. | Lub. Oil (kg) | Fuel Oil (kg) | Jacket CW (kg) | SAC Freshwater (kg) |

|------|---------------|---------------|----------------|---------------------|

| 5    | 1 300         | 10            | 675            | 410                 |

| 6    | 1 500         | 10            | 800            | 410 / 510           |

| 7    | 1 700         | 10            | 925            | 540 / 510           |

| 8    | 1 900         | 10            | 1 050          | 710                 |

**Air Filtration Requirements (Table 4-10)**

Dust concentration &gt; 0.5 mg/m³ requires inertial separator + oil-wetted filter to prevent accelerated piston-ring and liner wear.

**Exhaust-Gas Waste Gate (Delta Bypass Tuning – DBT)**

Closed below 50 % power (increases scavenge pressure); opens above 50 % to raise exhaust temperature by ~20 °C at 70 % power and increase steam production.

## 7.3 Wear Limits &amp; Torque Values (WinGD X62-B Maintenance Manual)

**Selected Clearances (0330-1/A1 – excerpt)**

- Thrust bearing axial clearance: nominal 0.4–0.65 mm, max 1.0 mm.

- Main bearing vertical clearance: nominal 0.25–0.55 mm, max 0.75 mm.

- Crosshead guide radial clearance: nominal 0.07–0.19 mm, max 0.2 mm.

- Piston-ring vertical clearance (top): nominal 0.40–0.48 mm, max 0.60–0.80 mm.

**Torque Values (0352-1/A1 – excerpt)**

- Elastic studs (M76×6): 1 500 bar hydraulic tension + 396 Nm.

- Exhaust-valve housing nuts (M72): 1 500 bar + 80°.

- Fuel-pump cover bolts: 100 Nm → 300 Nm → 480 Nm (three stages).

**Piston-Ring Chrome-Ceramic Wear (3425-1/A1)**

Measure with Permascope MP0. Minimum remaining layer: top ring 0.05 mm, middle/bottom 0.02 mm. Replace immediately if below limits.

## 7.4 Fluids &amp; Lubricants (MTU Series 2000/4000 – A001072/08E)

**Approved Engine Oils (SAE 40 only – no multigrades)**

- mtu GEO BG Power B2L, GEO NG Power X2L, GEO NG Power X3L (and listed alternatives).

**Used-Oil Analysis Limits (Table 2 – Series 4000 excerpt)**

- Viscosity at 100 °C: max 17.5 mm²/s, min 12.5 mm²/s (new-oil value +30 %).

- TBN: min 2.5 mg KOH/g and &gt; TAN (new-oil value –60 %).

- Water: max 0.2 % vol.

- Iron (Fe): max 30 mg/kg.

- Silicon (Si): max 15 mg/kg (natural-gas operation only).

**Cold-Corrosion Prevention**

Maintain jacket-cooling-water outlet ≥ 50 °C (warm engine) and cylinder-lube feed rate per sulphur content. Sample every 250 operating hours.

## 7.5 Latest MAN B&amp;W Service Letters (April 2025 Compilation – Selected)

- SL2025-766: New ALCU 2R Alpha Lubricator control unit.

- SL2024-758: Exhaust-valve spindle disc burn-off measurement tool.

- SL2024-756: PMI sensor calibration requirements.

- SL2023-741: Biofuel operation.

- SL2022-726: Service experience with 2020-compliant low-sulphur fuels.

- SL2021-714: Low-load operation (5–40 % load).

- SL2020-692: LDCL cooling-system update for very-low-sulphur fuels.

## 7.6 MTU Series 4000 Marine Solution Guide – Key Data

**Power Range (Marine Propulsion)**

- Series 2000: 720–1 939 kW (966–2 600 bhp).

- Series 4000: 746–4 300 kW (1 000–5 780 bhp).

**Emission Compliance**

- IMO Tier II / Tier III (SCR or EGR).

- EPA Tier 3r/4, EU Stage V, China C1/C2.

- Hybrid PropulsionPack option for battery-integrated silent running.

---

## Recommended Official PDFs to Download

Place these authoritative documents directly inside `core\_knowledge/marine\_main\_engine/` alongside the Markdown files.

1. **Everllence (MAN Energy Solutions) G60ME-C10.5 Project Guide** (latest edition, October 2025)

Direct download: https://man-es.com/applications/projectguides/2stroke/content/printed/G60ME-C10\_5.pdf

2. **WinGD RT-flex58T-E Marine Installation Manual** (Issue 2020-06)

Direct download: https://wingd.com/media/unrhs5eg/mim\_wingd\_rt-flex58t-e\_20200622.pdf

3. **WinGD X62-B Maintenance Manual** (Issue 2022-01)

Direct download: https://wingd.com/media/30gj0512/mm\_wingd-x62-b\_2022-01.pdf

4. **MTU Series 4000 Marine &amp; Offshore Solution Guide**

Direct download: https://www.mtu-solutions.com/content/dam/mtu/download/applications/commercial-marine/16120032\_MTU\_SolutionGuide\_Marine.pdf

5. **MTU Fluids and Lubricants Specifications (A001072/08E)** – applicable to all Series 2000/4000 marine engines

Direct download: https://www.mtu-solutions.com/content/dam/mtu/download/technical-info/betriebsstoffvorschrift\_en/A001072\_08E.pdf

6. **MAN B&amp;W Two-Stroke Service Letters – Table of Contents** (latest compilation)

Direct download: https://www.man-es.com/docs/default-source/man-primeserv/service-letters---table-of-content/2-stroke-table-of-content-temp.pdf

**End of Document**

Save this entire file as `marine\_main\_engine\_complete.md` inside `core\_knowledge/marine\_main\_engine/`. Convert to PDF using Pandoc, Microsoft Word, or any Markdown-to-PDF tool.