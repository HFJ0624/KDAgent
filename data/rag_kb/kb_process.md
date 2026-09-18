# SWaT Process Flow Knowledge

This file describes the overall process flow of the SWaT (Secure Water Treatment) testbed.
It is written as **general background knowledge only**. It must NOT be interpreted
as the ground truth of any specific case.

## Overview
SWaT is a six-stage water treatment process testbed (P1 through P6).
It contains continuous sensors (flow, level, pressure, analyzer) and discrete actuators
(motorized valves, pumps, ultraviolet unit).

## Stage description

### P1 — Raw water intake
- Sensors: FIT101 (flow), LIT101 (level of raw water tank)
- Actuators: MV101 (motorized inlet valve), P101/P102 (raw water pumps)
- Purpose: intake raw water from source and fill the raw water tank.

### P2 — Chemical dosing
- Sensors: AIT201/AIT202/AIT203 (analyzers for pH/conductivity/turbidity), FIT201 (flow)
- Actuators: MV201, P201~P206 (dosing pumps/valves)
- Purpose: inject chemical agents (coagulant, pH adjuster, disinfectant) based on water quality.

### P3 — Ultra-filtration / Membrane
- Sensors: DPIT301 (differential pressure), FIT301 (flow), LIT301 (level)
- Actuators: MV301~MV304, P301/P302
- Purpose: filter particles through membrane modules. Pressure differential is the key indicator of membrane fouling.

### P4 — UV treatment
- Sensors: AIT401/AIT402, FIT401, LIT401
- Actuators: P401~P404, UV401 (disinfection unit)
- Purpose: ultraviolet disinfection.

### P5 — Reverse Osmosis (RO)
- Sensors: AIT501~AIT504 (product water quality), FIT501~FIT504 (reject/permeate flows), PIT501~PIT503 (pressure)
- Actuators: P501/P502 (high-pressure pumps)
- Purpose: desalination. Pressure and concentrate flow are critical indicators of RO membrane condition.

### P6 — Product water delivery
- Sensors: FIT601 (delivery flow)
- Actuators: P601~P603 (delivery pumps)
- Purpose: deliver finished water to the distribution network.

## General cause–effect heuristics
- A sudden drop in **FIT101** is usually caused by closure of **MV101** or trip of **P101/P102**.
- A sustained increase in **LIT101** with decreasing outflow indicates **P101/P102** (inlet pumps) are running while **MV101** or downstream pumps are closed.
- A gradual rise in **DPIT301** indicates membrane fouling in P3; a sudden spike usually indicates a valve/PV rupture.
- Deviations of **AIT*** (analyzers) without corresponding actuator changes are often downstream symptoms rather than root causes.
- **Motorized valves (MV*)** and **pumps (P*)** are state variables; when their state does not match the flow/level response, the actuator is often the root cause.
