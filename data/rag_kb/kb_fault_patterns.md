# Generic Fault Patterns in SWaT

This file lists **generic** fault patterns observed in SWaT.
It is background knowledge only. It must NOT be treated as the ground truth of any specific case.

## 1. Sensor bias / drift
- **Symptoms**: A continuous variable (FIT*, LIT*, AIT*, PIT*, DPIT*) slowly drifts away from its expected value while other related signals remain normal.
- **Typical causes**: Calibration drift, fouled electrode (for AIT), clogged impulse line (for PIT/DPIT).
- **Stage hints**: Most common in P2 (AIT*) and P5 (PIT*, AIT*).

## 2. Sensor stuck / frozen
- **Symptoms**: A continuous variable stays at a constant value for a long time, despite process changes elsewhere.
- **Typical causes**: Defective transmitter, broken wiring, controller holding output.
- **Stage hints**: Can happen anywhere; most noticeable when FIT* or LIT* freezes during active flow changes.

## 3. Actuator stuck
- **Symptoms**: A state variable (MV*, P*, UV*) remains open/closed or on/off while the commanded state changes, or flow/level signals suggest a different operating condition.
- **Typical causes**: Mechanical jam, solenoid fault, motor burnout, controller command lost.
- **Stage hints**: MV101 (P1), P301/P302 (P3), P501/P502 (P5) are high-impact actuators.

## 4. Flow blockage
- **Symptoms**: FIT* drops abnormally; LIT* on the upstream side rises; DPIT* may increase.
- **Typical causes**: Strainer clog, membrane fouling (P3, P5), valve partially closed.
- **Stage hints**: DPIT301 (P3) and PIT501/PIT502 (P5) are the primary indicators.

## 5. Leakage
- **Symptoms**: LIT* on the upstream tank decreases faster than expected; downstream FIT* is lower than expected; pressure (PIT/DPIT) may drop.
- **Typical causes**: Burst pipe, leaking valve seal, open drain.

## 6. Inverter / VFD fault
- **Symptoms**: A pump state (P*) shows "on" but the corresponding FIT* does not respond, or responds weakly.
- **Typical causes**: Variable-frequency-drive fault, motor overheating, phase loss.

## 7. Chemical dosing anomaly
- **Symptoms**: AIT201~AIT203 (P2) deviate from their setpoints; downstream AIT401/AIT50* may also drift.
- **Typical causes**: Dosing pump (P201~P206) fault, MV201 misposition, exhausted chemical.
- **Stage hints**: Usually a *secondary* effect rather than a root cause, unless P2 actuators are the direct cause.

## 8. Membrane fouling (P3 / P5)
- **Symptoms**: DPIT301 (P3) or PIT501/PIT502 (P5) rises gradually; FIT301 / FIT501~FIT503 may drop.
- **Typical causes**: Accumulated particulates, chemical scaling, biological growth.
- **Stage hints**: Chronic trend; not a sudden change.

## 9. UV disinfection failure (P4)
- **Symptoms**: UV401 state abnormal; AIT401/AIT402 (chlorine/UV absorbance) changes unexpectedly.
- **Typical causes**: UV lamp aging, ballast fault, low flow through UV unit.

## General reasoning rules
1. Always distinguish **state variables** (MV*, P*, UV*) from **continuous measurements** (FIT*, LIT*, AIT*, PIT*, DPIT*). State variables are the *control*; continuous variables are the *response*.
2. When multiple continuous variables change simultaneously along a chain, trace back to the nearest state variable in the upstream stage.
3. Analyzers (AIT*) are rarely the root cause on their own; they usually reflect a disturbance originated elsewhere.
4. A mismatch between a state variable (e.g. MV101 = open) and its expected consequence (e.g. FIT101 should be high) is a strong indicator that the state variable itself (or the actuator behind it) is the root cause.
