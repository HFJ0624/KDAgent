# Industrial RCA Rules for SWaT RAG

This file contains general industrial reasoning rules for root cause analysis. It should be stored under `data/rag_kb/` and used as RAG background knowledge. Do not include any case-level ground truth.

## 1. Root Cause vs Downstream Effect

A variable with the highest anomaly score is not necessarily the root cause. It may be a downstream affected variable. A stronger root-cause candidate should explain why multiple other variables also become abnormal.

When choosing between a state actuator and a downstream continuous sensor:

- If the actuator state is unexpected and the downstream sensor changes consistently afterward, the actuator is more likely to be the root cause.
- If the continuous sensor is abnormal while related actuators and upstream variables remain normal, the sensor itself may be the root cause.

## 2. Continuous Sensor Reasoning

For continuous variables such as FIT, LIT, AIT, PIT, and DPIT, analyze:

- sustained residual deviation;
- sudden jump or drop;
- trend mismatch between raw and reconstructed values;
- high anomaly score over multiple time steps;
- consistency with related pump and valve states.

Typical interpretation:

- Flow anomaly may be caused by valve state, pump state, blockage, pressure change, or flow sensor fault.
- Level anomaly may be caused by imbalance between inflow and outflow, pump behavior, valve behavior, or level sensor fault.
- Pressure anomaly may be caused by pump behavior, blockage, valve configuration, or pressure sensor fault.
- Water-quality anomaly may be caused by upstream dosing, filtration, reverse osmosis, or analyzer sensor fault.

## 3. State Actuator Reasoning

For state variables such as MV, P, and UV, analyze:

- unexpected open/close or on/off switching;
- abnormal state persistence;
- mismatch between actuator state and downstream continuous variables;
- whether a state change can physically explain flow, level, pressure, or quality anomalies.

Typical interpretation:

- Valve abnormality often affects flow and downstream tank level.
- Pump abnormality often affects downstream flow, pressure, and sometimes tank level.
- UV unit abnormality should be interpreted with flow and treatment-stage context.

## 4. Common Cause-Effect Patterns

### Valve-related pattern

If a valve closes unexpectedly, related flow may drop and upstream/downstream level may change. If the valve opens unexpectedly, flow may increase or pressure/level balance may shift.

### Pump-related pattern

If a pump stops unexpectedly, downstream flow may decrease, pressure may change, and upstream tank level may rise. If a pump turns on unexpectedly, downstream flow or pressure may increase.

### Level-related pattern

If a level sensor shows abnormal residual but inflow and outflow actuators are consistent, the level sensor itself may be suspicious. If valve or pump states change first, the level variable may be downstream evidence.

### Flow-related pattern

Flow anomalies should be interpreted with valve state, pump state, and pressure. A flow sensor with high residual may be root cause only when related actuators cannot explain the deviation.

### Pressure-related pattern

Pressure anomalies may result from pump operation, valve closure, membrane/filter blockage, or pressure sensor issues. Pressure variables often provide evidence for pump/valve/filter-related root causes.

### Water-quality pattern

AIT anomalies may reflect true process changes or sensor faults. They should be interpreted with upstream dosing, filtration, reverse-osmosis behavior, and flow/pressure conditions.

## 5. Candidate Selection Rule

The final predicted root cause must be selected only from the Top-10 candidate list provided in the current case.

Do not invent variables outside the candidate list.

If evidence is weak, still choose the most plausible candidate from the Top-10 and lower the confidence.

## 6. Explanation Requirements

A good RCA explanation should include:

1. why the selected variable is abnormal;
2. how its variable type affects the interpretation;
3. how it can explain other abnormal candidates;
4. why high-score alternatives may be downstream effects;
5. the uncertainty of the decision.
