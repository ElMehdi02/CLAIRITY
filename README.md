
# 🫁 AuraDx — Smart Airway Seal Monitoring

### Nucleate Pittsburgh BioHack 2026 | RespAir Medical Devices Challenge

**AuraDx** is a proposed smart monitoring system designed to help healthcare professionals detect possible breathing tube seal problems early. Our solution combines multiple sensors, data analysis, and a real-time alert system to help nurses respond when something goes wrong.

![AuraDx Overview](images/AuraDx_%20Smart%20Airway%20Monitoring%20Pipeline-1.png)

---

## 1. The Problem

Patients who cannot breathe on their own may need a mechanical ventilator to help them breathe.

A ventilator connects to a breathing tube called an **endotracheal tube (ETT)**. This tube has a small inflatable balloon, called a **cuff**, that helps create a seal inside the patient's airway.

### What can go wrong?

Sometimes, the cuff does not create a proper seal, or the seal becomes weaker over time.

When this happens:

1. Air may leak around the breathing tube.
2. Saliva and other secretions may accumulate above the cuff.
3. Small amounts of these secretions may pass into the lungs. This is called **microaspiration**.
4. Microaspiration can contribute to lung infections, including ventilator-associated pneumonia (VAP).

### Why is this important?

Healthcare professionals need to know when a patient's breathing tube may not be properly sealed.

Although ventilators can detect certain air leaks, these measurements may not identify every problem involving the cuff or fluid passing around it.

Our goal is to provide additional information that helps nurses identify possible seal problems earlier.

---

## 2. Our Proposed Solution

We propose developing **AuraDx**, a smart monitoring system that combines a sensor-equipped breathing tube concept with software that analyzes the collected data.

Instead of relying on only one measurement, AuraDx would use multiple sensors to monitor different conditions around the breathing tube.

The system would:

- Monitor possible fluid accumulation above the cuff.
- Measure changes in fluid acidity.
- Monitor cuff pressure.
- Analyze ventilator airflow and pressure readings.
- Combine sensor information to identify possible problems.
- Display the readings on a real-time dashboard.
- Send alerts to nurses when the system detects a possible seal problem.

Our long-term goal is to connect AuraDx with hospital monitoring software so healthcare professionals can receive alerts through their existing systems.

![AuraDx Monitoring Workflow](images/Connected%20Respiratory%20Monitoring%20Workflow.png)

---

## 3. Why We Chose These Sensors

Our team explored different sensor technologies and selected several possible approaches.

### Sensor 1 — Capacitive Sensor

**Purpose:** Detect possible fluid accumulation above the breathing tube cuff.

A capacitive sensor measures changes in electrical capacitance. When fluid accumulates near the sensor, its readings may change.

**Why we proposed it:**

- It could provide an early indication of fluid buildup.
- It could continuously monitor changes over time.
- It could help identify conditions associated with microaspiration risk.

However, fluid accumulation does not automatically mean the cuff is leaking.

### Sensor 2 — pH Sensor Above the Cuff

**Purpose:** Measure the acidity or alkalinity of fluid accumulating above the cuff.

Different fluids can have different pH values.

**Why we proposed it:**

- It could provide additional information about accumulated secretions.
- It could help identify unusual changes in fluid properties.
- It could work alongside the capacitive sensor.

A pH measurement alone cannot reliably identify the fluid's origin or determine whether an infection is present.

### Sensor 3 — pH Sensor Below the Cuff

**Purpose:** Explore whether changes in fluid chemistry below the cuff could provide evidence of possible fluid passage.

**Why we proposed it:**

- It could provide information from another location around the breathing tube.
- It could help researchers investigate possible changes associated with fluid passage.

This sensor is currently a research concept. Placing sensors below the cuff introduces additional medical and safety challenges.

For our initial prototype, we plan to simulate this sensor rather than place it inside a patient's airway.

### Sensor 4 — Cuff Pressure Sensor

**Purpose:** Monitor pressure inside the inflatable cuff.

**Why we proposed it:**

- Changes in cuff pressure may indicate conditions that need attention.
- It provides another measurement to support the monitoring system.
- It can help the software detect unusual pressure changes.

Cuff pressure does not directly prove that the cuff has a complete seal.

### Sensor 5 — Ventilator Waveform Analysis

**Purpose:** Analyze airflow, airway pressure, and breathing volume measurements.

**Why we proposed it:**

- Air leaks can sometimes create changes in ventilator measurements.
- The software can analyze patterns over time.
- Combining ventilator data with sensor readings may help identify possible seal problems.

This is one of the main parts of our proposed software solution.

---

## 4. How AuraDx Will Work

Our system will have four main stages.

### Step 1 — Collect Sensor Data

The sensors will continuously measure different conditions around the breathing tube.

Examples include:

- Fluid accumulation signals
- pH measurements
- Cuff pressure
- Airflow
- Airway pressure
- Breathing volume differences

### Step 2 — Analyze the Data

A microcontroller will collect sensor measurements and send them to our software.

The software will use Python to:

- Process incoming measurements.
- Compare readings with previous values.
- Detect unusual changes.
- Combine information from different sensors.
- Identify possible sensor failures.
- Determine whether an alert should be generated.

### Step 3 — Display Results

We plan to develop a dashboard showing:

- Live sensor measurements
- Pressure and airflow graphs
- Possible leak warnings
- Sensor connection status
- Previous alerts
- Explanations of detected abnormalities

### Step 4 — Alert Healthcare Professionals

If the software identifies a possible seal problem, it will generate an alert.

Our proposed alert system includes:

- 🟢 **Green:** No abnormality detected.
- 🟡 **Yellow:** Unusual readings that need attention.
- 🔴 **Red:** Possible seal problem requiring prompt investigation.
- ⚪ **Sensor Fault:** Missing or unreliable sensor data.

These alert levels will need testing and validation.

In the future, we hope to integrate the system with hospital software so nurses can receive notifications directly.

This hospital integration is a proposed feature and has not yet been implemented.

---

## 5. Why We Want to Combine Multiple Sensors

One of our main ideas is to avoid depending on only one sensor.

For example:

A capacitive sensor might detect fluid accumulation, but that does not necessarily mean fluid has passed the cuff.

A pressure sensor might detect a pressure change, but that does not always mean the cuff has lost its seal.

By combining different measurements, our system may be able to provide more useful information.

We hope this approach will:

- Improve the reliability of warnings.
- Reduce unnecessary alerts.
- Detect unusual changes earlier.
- Help healthcare professionals understand why an alert occurred.

These benefits are goals that we still need to test.

---

## 6. Our Software Features

We plan to develop the following features.

### Real-Time Monitoring

Display incoming sensor readings and ventilator measurements.

### Multi-Sensor Analysis

Combine information from different sensors to detect possible seal problems.

### Smart Alert System

Generate warnings when abnormal patterns are detected.

### Sensor Failure Detection

Identify disconnected sensors, missing readings, or unexpected sensor behavior.

### Patient Baseline Monitoring

Compare new measurements with earlier readings to identify unusual changes.

### Alert History

Save detected events with timestamps and information about which sensors triggered the warning.

### Explainable Detection

Show healthcare professionals why the software generated an alert.

### AI and Machine Learning

If sufficient labeled data becomes available, we may use machine learning to identify patterns associated with possible leaks.

We will begin with simple detection rules before developing more advanced models.

---

## 7. What We Have Come Up With So Far

After discussing the RespAir challenge, our team developed the concept of a multi-sensor airway monitoring system.

Our proposed design includes:

1. A sensor-based system for monitoring conditions around the breathing tube cuff.
2. A combination of fluid, pH, pressure, and ventilator measurements.
3. Software that analyzes incoming sensor data.
4. A dashboard that displays measurements and warnings.
5. An alert system designed to notify healthcare professionals.
6. A future connection with hospital monitoring software.

Our main idea is to combine hardware and software into one monitoring system.

Rather than waiting for a single measurement to indicate a problem, we want AuraDx to examine multiple signals and provide an understandable warning.

---

## 8. Our Findings and Results

### Research Findings

During our initial research and discussions, we identified several important points:

- An inadequate breathing tube cuff seal may contribute to air leakage and microaspiration.
- Fluid accumulation above the cuff can occur without a confirmed seal failure.
- Cuff pressure alone does not fully describe the quality of the seal.
- Ventilator airflow and pressure measurements may help identify possible leaks.
- Combining different sensors could provide additional information.
- Medical monitoring systems must consider false alarms and sensor failures.

These findings helped us develop the AuraDx concept.

### Prototype Results

**Current status: Planning and development**

We have developed our proposed system design and identified the sensors and software features we want to investigate.

We have not yet completed experimental testing or clinical validation.

Once our prototype is working, we plan to measure:

| Test | What We Want to Evaluate |
|---|---|
| Normal conditions | Does the system avoid unnecessary alerts? |
| Fluid accumulation | Can the sensor detect changes in fluid levels? |
| Possible cuff leak | Can the system identify abnormal patterns? |
| Sensor failure | Can the software detect missing or invalid readings? |
| Detection time | How quickly does the system generate an alert? |
| False alarms | How often does it generate an incorrect warning? |
| Multiple sensors | Does combining sensors improve detection? |

We will add our actual results after completing these experiments.

---

## 9. Technologies We Plan to Use

### Hardware

- Capacitive fluid sensor
- pH sensors
- Pressure sensor
- ESP32 or similar microcontroller
- Simulated ventilator signals or an approved benchtop test setup

### Software

- Python for data processing
- pandas and NumPy for calculations
- Streamlit or another web framework for the dashboard
- Plotly for real-time graphs
- Machine learning libraries if needed
- GitHub for collaboration

The final technologies may change as we test the prototype.

---

## 10. Our Next Steps

1. Finalize the sensor selection.
2. Develop a safe benchtop testing setup.
3. Connect the available sensors to a microcontroller.
4. Collect sample measurements.
5. Build the Python data analysis system.
6. Create the monitoring dashboard.
7. Implement the alert system.
8. Test normal conditions and simulated leak scenarios.
9. Evaluate detection accuracy and false alarms.
10. Demonstrate a simulated nurse notification workflow.

---

## 11. Our Long-Term Vision

Our long-term vision is to create a monitoring system that can support healthcare professionals caring for patients who require mechanical ventilation.

We hope AuraDx could eventually help identify possible airway seal problems earlier and provide nurses with clearer information.

Before any clinical use, the device and software would require appropriate engineering testing, safety evaluation, clinical validation, and regulatory review.

**AuraDx is currently a research and hackathon prototype concept. It is not an approved medical device and should not be used for patient monitoring or clinical decisions.**

---

## 12. Project Information

**Project:** AuraDx — Smart Airway Seal Monitoring

**Event:** Nucleate Pittsburgh BioHack 2026

**Challenge Sponsor:** RespAir Medical Devices

**Track:** MedTech — Hardware & Biosensors

**Project Focus:** Medical Sensors, Data Science, Real-Time Monitoring, and Healthcare Software

---

*Developed as part of Nucleate Pittsburgh BioHack 2026.*