# Interview Preparation - Predictive Maintenance Project

**Golden rule:** always say the data is *simulated* and the thresholds are *demonstration values*. Honesty impresses interviewers.

## A. 30-second explanation
"I built a Python condition-monitoring system for a rotating machine like a motor or pump. It takes temperature, vibration, RPM and current, classifies the machine as normal, warning or fault, gives a health score and a maintenance recommendation, and shows it in a dashboard. I used simulated data because I had no physical sensors, so it's a learning project, but it follows the same logic a real condition-monitoring system uses."

## B. 1-minute explanation
"Unplanned breakdowns of motors and pumps are expensive, so I wanted to understand predictive maintenance. I simulated 6,000+ readings with gradual degradation - bearing wear, imbalance, misalignment, overload, overspeed. I calculated moving averages and rates of change, then built transparent threshold rules: normal, warning or fault, with possible causes and recommended actions. I also built a 0-100 health score and trained a Random Forest. On unseen simulated runs it reached about 98% accuracy, but I'm clear that this is simulated data, so it proves the pipeline works, not real-world accuracy. A Streamlit dashboard lets me inject faults and watch the system react."

## C. 3-minute detailed explanation
1. **Problem:** reactive maintenance = downtime; fixed-interval maintenance = wasted work. Condition data lets us act when needed.
2. **Data:** 7 scenarios x 4 runs x 220 one-minute samples = 6,160 rows. In each fault run a hidden severity rises from 0 to 1, and signals respond non-linearly because faults accelerate. Temperature has thermal lag, like real metal housings. Noise and load variation are added. Seed 42 makes it reproducible.
3. **Indicators:** 10-sample moving average (removes noise), rate of change (trend), rolling RMS of vibration.
4. **Rules:** each parameter has a warning and fault threshold (project-defined, editable). Overall status = worst parameter. Two or more abnormal parameters = combined condition, escalate to qualified staff.
5. **Health score:** each parameter gets a sub-score (100 at nominal, 79 at warning, 59 at fault). Health = lowest sub-score minus 5 per additional abnormal parameter - weakest link.
6. **ML:** Random Forest on sensor + trend features. I split by whole operating runs, not random rows, to avoid leakage. Test accuracy 0.980, macro precision 0.980, macro recall 0.978; group cross-validation 0.977. The rule-based detector scored 0.863 on the same test runs, mostly because it flags warnings later than the labels.
7. **Limits:** simulated, single RMS value instead of spectra, not a safety system, risk indicator is not Remaining Useful Life.
8. **Next step:** connect real sensors, add FFT analysis, validate on public bearing datasets.

## D & E. Ten likely questions with answers

**1. Why did you choose vibration?**
It is the most sensitive early indicator for rotating machinery. Imbalance, misalignment, looseness and bearing wear all change vibration before temperature or current react noticeably.

**2. Why is temperature important?**
Friction, poor lubrication, bad cooling and overload all produce heat. It reacts slowly because the machine has thermal mass, so it confirms a developing problem rather than giving the earliest warning.

**3. What is RMS vibration?**
Root-mean-square: square the signal, average it, take the square root. It gives one number for the effective vibration energy, because a plain average of an oscillating signal would be about zero. Unit here: mm/s (velocity).

**4. Predictive vs preventive maintenance?**
Preventive = service on a fixed schedule (every X hours) regardless of condition. Predictive = service based on measured condition and trends, so you act when the data shows it is needed. Reactive = fix after failure.

**5. Why Random Forest?**
Good for small tabular data, handles different units without scaling, resists overfitting, needs little tuning, and gives feature importance I can explain. Deep learning would be unnecessary and hard to justify with this data.

**6. How did you generate the data?**
With a Python simulator: baseline values plus noise, then a hidden severity ramp per fault scenario that shifts vibration, temperature, current and RPM in physically plausible directions (e.g. overload raises current and temperature and slightly lowers RPM). It is simulated, not measured, and I say so everywhere.

**7. How does fault detection work?**
Each parameter is compared with a warning and fault threshold. Status is the worst level. If two or more parameters are abnormal it is a combined condition and the recommendation is to follow the safety procedure and escalate. Thresholds are demonstration values and editable.

**8. What happens when vibration increases?**
Vibration sub-score falls, health score drops, status moves NORMAL to WARNING to FAULT, the diagnosis says "possible imbalance, misalignment or bearing abnormality", and the recommendation is to inspect alignment, balance and bearings. The trend slope also raises the maintenance risk level.

**9. What are the limitations?**
Simulated data; high accuracy is partly because labels and signals come from the same simulator; one RMS value instead of frequency spectra; four signals cannot prove root cause; thresholds are not from a standard; no true RUL; not a safety system.

**10. How would you connect it to real sensors?**
Accelerometer (vibration), thermocouple/RTD (temperature), tachometer or drive output (RPM), current transformer/clamp (current), through a DAQ, PLC/Modbus or microcontroller. Stream readings into the same pipeline, set thresholds from the OEM manual, ISO vibration guidance and a measured baseline of the healthy machine, and validate against real inspection results.

## F. Terms to know
- **Condition monitoring:** measuring machine health parameters continuously or periodically.
- **Predictive maintenance:** maintenance decided by condition and trend.
- **Preventive maintenance:** maintenance on a fixed time/usage schedule.
- **Rotating machinery:** motors, pumps, fans, compressors, gearboxes.
- **RMS:** root-mean-square, effective magnitude of an oscillating signal.
- **Imbalance:** uneven mass distribution on the rotor; causes vibration at 1x running speed.
- **Misalignment:** shafts not collinear; often shows 2x speed vibration and heat.
- **Bearing fault:** wear/damage/poor lubrication; raises vibration and temperature.
- **Overload:** load beyond rating; high current and temperature, lower speed.
- **Overspeed:** speed above allowed limit; mechanical stress risk.
- **Thermal inertia:** temperature changes slowly because of heat capacity.
- **Moving average:** average of last N samples; smooths noise.
- **Rate of change / trend:** how fast a value rises per unit time.
- **Health score:** single 0-100 summary of condition.
- **Random Forest:** many decision trees voting together.
- **Accuracy / precision / recall:** share correct overall / share of predicted faults that were real / share of real faults caught.
- **Confusion matrix:** table of true vs predicted classes.
- **Data leakage:** test information accidentally influencing training, giving inflated results.
- **Feature importance:** how much each input helped the model.
- **RUL (Remaining Useful Life):** estimated time to failure; needs run-to-failure data, which I do not have.
- **FFT:** converts a vibration signal into frequencies to identify fault types.
- **FSE (Field Service Engineer):** installs, diagnoses, maintains and repairs equipment at customer sites.

## 14. Resume bullets (truthful, ATS-friendly)
- Developed a Python-based condition-monitoring and predictive-maintenance prototype for rotating machinery (motor/pump) using simulated temperature, vibration, RPM and motor-current data, with a Streamlit dashboard for real-time fault diagnostics and maintenance recommendations.
- Implemented rule-based fault detection, a 0-100 machine health score and trend indicators (moving average, rolling RMS vibration), classifying NORMAL/WARNING/FAULT states and mapping them to troubleshooting and root-cause inspection actions.
- Trained and validated a Random Forest classifier with leakage-safe, run-wise splitting (98% accuracy on held-out simulated runs; documented as simulation-only) and documented the system for technical handover.
