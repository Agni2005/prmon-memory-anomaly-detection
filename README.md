This project explores anomaly detection on memory time-series data generated using prmon.

The data is intentionally simple. The anomaly is implemented as a sustained increase in allocated memory (500 → 650) using mem-burner. This step-like change was chosen
deliberately to isolate how detection methods behave under a clear regime shift. Starting with a controlled setup makes it easier to interpret results before introducing 
noise, drift, or more complex patterns.

Two statistical approaches were used:

Rolling Z-score compares each point to recent window statistics. It adapts over time, which makes it suitable for detecting sudden spikes. 
However, it adapts to sustained shifts and therefore performs poorly for step anomalies.

Frozen baseline Z-score computes statistics from the initial normal segment and uses that fixed reference for all future points. 
It does not adapt, which makes it effective for detecting sustained structural changes when a clean baseline is available.

Machine learning methods were not used because the dataset is small and intentionally structured. The anomaly is a clear mean shift, 
and statistical methods are sufficient and more interpretable for this controlled warm-up.

AI assistance (ChatGPT) was used for structuring documentation, refining explanations, and minor code formatting. 
The experimental setup, anomaly injection, implementation logic, and evaluation were designed and executed independently.
