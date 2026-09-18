# Sensor / Actuator Relations

This file documents the relationship between SWaT sensors and actuators.
**This is generic engineering knowledge, NOT the ground truth of any case.**

## Variable name prefix dictionary

| Prefix | Meaning                           | Typical type | Typical unit / semantics |
|--------|-----------------------------------|--------------|---------------------------|
| FIT    | Flow Indicator / Transmitter      | continuous   | flow                      |
| LIT    | Level Indicator / Transmitter     | continuous   | level                     |
| AIT    | Analyzer Indicator / Transmitter  | continuous   | analyzer (pH, conductivity, turbidity, chlorine) |
| PIT    | Pressure Indicator / Transmitter  | continuous   | pressure                  |
| DPIT   | Differential Pressure Indicator / Transmitter | continuous | differential_pressure |
| MV     | Motorized Valve                   | state        | state (open/closed)       |
| P      | Pump                              | state        | state (on/off)            |
| UV     | Ultraviolet disinfection unit     | state        | state                     |

## Stage relations

### P1 (Intake)
- FIT101 is the inlet flow sensor. It responds to MV101 (inlet valve) and P101/P102 (inlet pumps).
- LIT101 is the raw tank level. It is governed by the balance of FIT101 inflow and P101/P102 outflow.

### P2 (Dosing)
- AIT201, AIT202, AIT203 measure water quality after chemical dosing. They respond to MV201 and P201~P206 (dosing pumps).
- FIT201 measures the outflow of P2.

### P3 (Ultra-filtration)
- DPIT301 is the transmembrane differential pressure. It is the most direct indicator of membrane fouling.
- FIT301 is the filtrate flow.
- LIT301 is the membrane feed tank level.
- MV301~MV304 control the membrane feed/retentate paths.
- P301/P302 are the membrane feed pumps.

### P4 (UV)
- AIT401, AIT402 measure water quality after UV treatment.
- FIT401 measures the flow through the UV unit.
- LIT401 is the UV feed tank level.
- P401~P404 are the transfer pumps; UV401 is the UV disinfection unit.

### P5 (RO)
- AIT501~AIT504 measure RO permeate water quality.
- FIT501~FIT504 measure permeate and concentrate flows.
- PIT501~PIT503 measure RO feed / concentrate / permeate pressures.
- P501/P502 are the high-pressure pumps for RO.

### P6 (Delivery)
- FIT601 is the final delivery flow.
- P601~P603 are the delivery pumps.

## Coupling hints
- Flow sensors (FIT*) and level sensors (LIT*) are coupled through pump/valve states.
- Pressure sensors (PIT*, DPIT*) respond to downstream valve positions and filter condition.
- Analyzers (AIT*) are usually the *result* of a disturbance, rarely the root cause.
- State variables (MV*, P*, UV*) are the most common root-cause candidates because they directly control the process.

## Detailed Variable Relationships

### P1 (Intake)
- MV101 controls raw water inflow. MV101 abnormal closure may reduce FIT101 and affect LIT101.
- FIT101 measures raw water inflow. Changes in FIT101 may affect LIT101 tank level.
- LIT101 reflects P1 tank level. Abnormal LIT101 may indicate level sensor fault or inflow/outflow imbalance.
- P101 and P102 are pumps related to water transfer from P1 to downstream stages.

### P3 (Ultra-filtration)
- MV301, MV302, MV303, MV304 are P3 motorized valves. Their abnormal switching may affect P3 ultrafiltration flow and tank level.
- P301 and P302 are P3 pump state/control variables. Abnormal pump state may affect downstream flow, pressure, and delivery-stage variables.
- DPIT301 is the transmembrane differential pressure. It is the most direct indicator of membrane fouling.

### P4 (UV)
- LIT401 is related to P4 tank level. It can be affected by upstream P3 ultrafiltration flow and downstream pumping behavior.

### P6 (Delivery)
- P601, P602, P603 are P6 delivery/distribution pump variables. Their anomalies may be downstream responses to earlier-stage disturbances.

## Root Cause Reasoning Guidelines
- State variables such as pumps and valves often represent control actions, while continuous variables such as flow, level, and pressure are often system responses.
- A high anomaly score does not guarantee root cause. A root cause candidate should explain other abnormal variables.
- Root causes typically appear *before* symptoms in the time series. If a variable shows anomalies earlier than others, it is more likely to be the root cause.
- Control variables (MV*, P*) that change state abruptly are strong root cause candidates, as they represent deliberate or malfunctioning control actions.
