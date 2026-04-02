# MARINE INTEGRATED ALARM MONITORING AND CONTROL SYSTEM (AMS / IAMCS)

**Equipment:** Marine Integrated Alarm Monitoring and Control System (AMS / IAMCS) – (e.g., Kongsberg K-Chief, Wärtsilä NACOS Platinum, ABB 800xA, Valmet DNA)  
**Folder Name:** alarm_monitoring_system  

**Part 1 – Fundamentals & Physics Behind Operation**  

The marine Alarm Monitoring and Control System (AMS / IAMCS) functions as the vessel’s central nervous system, continuously acquiring, processing, and presenting real-time data from field sensors to enable safe machinery operation and rapid fault response. Physical parameters (temperature, pressure, level, flow, vibration) are transduced into electrical signals via resistance (PT100 RTDs), voltage (thermocouples), or current (4–20 mA transmitters). The 4–20 mA current loop is the dominant marine standard: the transmitter modulates loop current between 4 mA (live-zero, representing 0 % of scale) and 20 mA (100 % of scale). This current-driven signal is immune to line resistance drops over cable runs up to 1 000 m and provides built-in wire-break detection (0 mA = open circuit).  

Signal conditioning occurs in galvanically isolated I/O modules where analogue-to-digital conversion (typically 12- or 16-bit resolution) maps the 4–20 mA range into digital counts. Digital communication uses fieldbuses: CAN-bus employs differential voltage (CAN_High / CAN_Low) with CSMA/CR arbitration, guaranteeing deterministic delivery of high-priority alarms (e.g., main-engine slowdown) even under heavy bus load. Ethernet-based system buses use MRP or PRP redundancy protocols to achieve <100 ms failover on single cable failure. Modbus RTU/TCP (RS-485 or Ethernet) provides third-party integration.  

Redundancy physics (ABS Rules Part 4 Chapter 9, DNV Pt.4 Ch.9, IACS UR E22/E26/E27) mandate hot-standby CPU pairs with heartbeat monitoring: loss of the master heartbeat (<100 ms) forces automatic switchover. Dual independent 24 V DC supplies (main switchboard + battery UPS) ensure uninterrupted operation. Digital inputs are normally-closed wherever possible, so a wire break generates an immediate “loop fault” alarm rather than a hidden failure.  

Abnormal behaviour originates from violation of these principles: a 4–20 mA loop falling below 3.8 mA triggers “sensor failure” because live-zero is lost; CAN-bus reflection from missing 120 Ω termination causes message collisions and “node offline” alarms; power supply A/B imbalance or UPS battery discharge leads to CPU watchdog reset and total system reboot. All class rules (ABS, DNV, LR) require functional independence of safety-critical alarms from control systems, ensuring that an AMS fault cannot inhibit an independent engine shutdown signal.  

**Part 2 – Major Components & Auxiliary Systems**  

**2.1 Field Input/Output (I/O) Modules**  
Located in distributed Local Operating Panels (LOPs) near machinery.  
- Analog Input (AI): Accepts PT100/PT1000, thermocouples (Type K/J), 4–20 mA, 0–10 V. Galvanic isolation per channel (≥1 500 V), 12/16-bit A/D, loop-power supply (24 V).  
- Digital Input (DI): Opto-isolated 24 V DC dry contacts; supports normally-open/closed with debounce filtering.  
- Digital Output (DO): Relay (5 A) or transistor outputs for horns, beacons, solenoids.  
- Analog Output (AO): 4–20 mA for actuator positioning (e.g., fuel-rack or valve control).  

**2.2 Process Control Units (PCUs) / Controllers**  
- CPU Module: Dual hot-standby processors (embedded Linux/QNX or PLC-based IEC 61131); executes alarm logic, trending, and HMI data distribution.  
- Communication Gateways: CAN-bus, Profibus, Modbus TCP/RTU, Ethernet (100 Mbps/1 Gbps).  
- Power Supply Unit (PSU): Dual-redundant 24 V DC → 5 V/3.3 V internal rails with diode-OR and battery backup.  

**2.3 Operator Stations (OS) and HMI**  
- Industrial PCs (IPC) with high-brightness, dimmable marine-grade monitors (Bridge, ECR, ECR wing).  
- SCADA/HMI Software: Graphical mimics, alarm lists, trend graphs, diagnostic pages (Kongsberg K-Chief, Wärtsilä NACOS Platinum, ABB 800xA, Valmet DNA).  
- Alarm Extension System: Duty alarm panels in cabins, mess, public spaces with audible/visual signals and group alarm repeaters.  

**2.4 Sensors and Field Devices**  
- Pressure: Piezo-resistive or capacitive 4–20 mA transmitters.  
- Temperature: PT100 (standard for linearity) or thermocouples.  
- Level: Float, capacitive, ultrasonic, or radar.  
- Vibration: Accelerometers with 4–20 mA or frequency output.  
- Critical safety sensors (overspeed, low lube-oil pressure) are independent and hard-wired to engine safety system (not routed through AMS).  

**2.5 Auxiliary Systems**  
- Uninterruptible Power Supply (UPS): Dual 24 V battery banks sized for ≥30 min autonomy (SOLAS II-1/42).  
- Network Infrastructure: Redundant Ethernet switches with MRP/PRP, fibre-optic backbone where required.  
- Data Logging: Event historian with VDR interface (IEC 61162-450 / NMEA).  

All components comply with ABS Part 4 Chapter 9, DNV Pt.4 Ch.9, IACS UR E22/E26/E27, and manufacturer project guides (Kongsberg K-Chief 600/700, Wärtsilä NACOS Platinum).  

**Part 3 – Normal Operation & Key Parameters**  

A healthy AMS is silent and invisible. All CPUs and I/O modules display steady green “Run” LEDs with rhythmic amber “Comm” flashing. The alarm list is clean (no active unacknowledged alarms). Network diagnostic pages show zero “Comm Loss” or “Node Offline”.  

**Typical Healthy Parameters**  

| Parameter                  | Healthy Range                          | Alarm Threshold                  |
|----------------------------|----------------------------------------|----------------------------------|
| I/O Loop Supply Voltage    | 18–26 V DC                             | <15 V (loop fault)               |
| 4–20 mA Analogue Signal    | 3.8–20.5 mA                            | <3.5 mA (open) / >21 mA (short) |
| CAN/Ethernet Network Load  | <40 % bandwidth                        | >70 % (congestion)               |
| CPU / IPC Temperature      | 30–55 °C                               | >75 °C (overheating)             |
| UPS Battery Float Voltage  | 26.5–27.5 V                            | <23 V (discharged)               |
| Heartbeat Interval         | <100 ms                                | >200 ms (failover imminent)      |

Functional priority (IMO Code on Alarms and Indicators A.1021(26) and class rules):  
1. Safety (Red) – immediate action (e.g., main-engine slowdown).  
2. Warning (Amber) – attention required.  
3. Status (Blue/White) – informational.  
4. System (Internal) – AMS self-diagnostics.  

**Part 4 – Common Faults, Symptoms & Root Causes**  

**4.1 Sensor / Loop Failure**  
Symptoms: “Sensor Failure” or “Range Error” on mimic (### or fixed value).  
Physics: Current loop <3.8 mA or >20.5 mA.  
Root Causes: Loose terminals, water ingress in junction box, failed transmitter diaphragm.  
Real-World Example: Intermittent “Low LO Press” alarms in heavy weather – water shorting signal to hull (ABS survey finding).  

**4.2 Communication Loss / Node Offline**  
Symptoms: Group of sensors (e.g., all cylinder temps) freeze; HMI shows “Node 5 Offline”.  
Physics: Loss of keep-alive telegram on CAN-bus or Ethernet.  
Root Causes: Vibration-induced connector failure, missing 120 Ω termination resistor, blown 24 V fuse on I/O rack.  
Real-World Example: Kongsberg K-Chief intermittent node dropout – improper CAN-bus termination causing reflections (Kongsberg field service bulletin).  

**4.3 Power Supply / UPS Failure**  
Symptoms: “Power Supply A Failure” alarm; possible system reboot during generator changeover.  
Root Causes: Tripped breaker, failed diode-bridge, discharged UPS batteries.  
Real-World Example: AMS blackout during main-generator transfer – UPS batteries unable to bridge 2-second dip (DNV incident report).  

**4.4 Nuisance / False Alarms**  
Symptoms: “High Bilge Level” flickering.  
Root Causes: Tank sloshing combined with insufficient delay timer in software.  

**Part 5 – Expert Troubleshooting Guide & Trade Tricks**  

**Safety Note:** Never inhibit safety-critical alarms while machinery is running. Always use permit-to-work and verify zero energy on field loops.  

**5.1 4–20 mA Loop Isolation (Most Important Minute Trick)**  
1. Disconnect field wires at I/O terminal.  
2. Inject 12 mA (50 % scale) with loop calibrator.  
3. If HMI shows exact 50 %, AMS module is healthy – fault is in cable or sensor.  
Trade Trick: No calibrator? Use 1.5 V battery + 100 Ω resistor for rough 15 mA test.  

**5.2 Heartbeat LED Diagnosis**  
- Fast flashing → heavy comms.  
- Slow steady → normal.  
- Solid ON/OFF → CPU hung.  
- Red → self-test fail (replace module).  

**5.3 Ground-Fault Hunting**  
1. Measure 24 V system to hull (must be floating).  
2. If 0 V on negative to hull, disconnect field cables one-by-one until fault clears. Last cable = culprit.  

**5.4 Software Force / Inhibit**  
- Inhibit during maintenance (e.g., fire-pump test) to prevent ship-wide alarms.  
- Force digital output only for beacon/horn test – never force shutdown signals.  

**5.5 Quick Network Checks**  
- Verify 120 Ω termination resistors at both CAN-bus ends.  
- Check Ethernet switch port LEDs and MRP ring status.  

**Part 6 – Maintenance & Preventive Checks**  

Refer to OEM manuals for detailed routine schedules and procedures (Kongsberg K-Chief Project Guide, Wärtsilä NACOS Platinum O&M, ABB 800xA / Valmet DNA service literature, and classification-approved maintenance plans). All intervals, torque values, and acceptance criteria must be taken directly from the vessel-specific approved OEM documentation.  

**Recommended Official PDFs to Download**  
1. ABS Rules for Building and Classing Marine Vessels 2018 – Part 4, Chapter 9: Control and Monitoring Systems – https://ww2.eagle.org/content/dam/eagle/rules-and-guides/archives/other/1000_marinevessels_2018/mvr-part-4-aug-18.pdf  
2. DNV Rules Pt.4 Ch.9 – Control and Monitoring Systems – https://civamblog.files.wordpress.com/2016/11/ts409.pdf  
3. IMO Resolution A.1021(26) – Code on Alarms and Indicators – https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/AssemblyDocuments/A.1021(26).pdf  
4. Kongsberg K-Chief 600 Marine Automation System Product Documentation – https://www.kongsberg.com/globalassets/kongsberg-maritime/km-products/product-documents/k-chief-600-marine-automation-system/  
5. Wärtsilä NACOS Platinum Integrated Automation & Navigation System Guide – https://www.wartsila.com/docs/default-source/marine-documents/events/kormarine/kormarine-2021/nacos-platinum-navigation-automation.pdf  
6. ABS Guidance Notes on Response Time Analysis for Programmable Electronic Alarm Systems – https://ww2.eagle.org/content/dam/eagle/rules-and-guides/current/design_and_analysis/303-gn-response-time-analysis-programmable-electronic-alarm-systems-2018/rta-gn-sept18.pdf  