# 🌞 Aditya-L1 Plasma Lab (Redesigned)

> India's first dedicated solar mission dashboard, transformed into a modern, high-performance Single Page Application (SPA). This laboratory tracks real-time solar wind plasma data from the Sun-Earth L1 Lagrange point, providing critical insights into space weather and solar dynamics.

🔗 **Live site**: https://vivek-shrivastav.github.io/Aditya-L1_Plasma_Lab/

---

## 🏗️ New Architecture (SPA)

The platform has been consolidated into a single-entry SPA for a faster, more cohesive user experience. All telemetry, visualizations, and scientific tools are accessible through a unified dashboard shell.

| View | Purpose |
|------|---------|
| **Home** | Mission context, live telemetry snapshot, and Sun-L1-Earth geometry. |
| **Dashboard** | Real-time monitoring of Density, Speed, Magnetic Field (Bx, By, Bz), and Temperature with auto-refresh. |
| **Visualizer** | 2D projection of the interplanetary environment, including IMF Clock Angle and Kp Index estimation. |
| **Explorer** | Science reference desk, instrument specifications, and physics glossary. |
| **Analysis Lab** | Scientific workspace with Spectral Analysis (PSD), Plasma Moments Calculator, and Event Detection. |

---

## 🛰️ Live Data Source

To ensure 24/7 reliability while the Aditya-L1 PRADAN pipeline is in transition, this dashboard currently fetches high-cadence live data from the **NOAA SWPC DSCOVR (Deep Space Climate Observatory)** satellite.

- **Orbit**: L1 Lagrange Point (same vantage point as Aditya-L1).
- **Update Frequency**: ~60 seconds.
- **Parameters**: Proton Density (cm⁻³), Bulk Speed (km/s), Proton Temperature (K), Magnetic Field (GSM Bx, By, Bz, Bt in nT).

---

## 🛠️ Technology Stack

- **Core**: Vanilla JavaScript (SPA Component Architecture)
- **Styling**: Modern CSS3 (Glassmorphism, Responsive Grid System)
- **Visuals**: Plotly.js for high-fidelity scientific charting
- **Animation**: CSS Keyframes + SVG for geometry and UI transitions
- **Deployment**: GitHub Pages (Static Hosting)

---

## 🔬 Scientific Foundations

The Analysis Lab and Visualizer modules implement standard space physics calculations:
- **Alfvén Speed**: Computed from the live local magnetic field and proton density.
- **IMF Clock Angle**: Projection of magnetic field By and Bz components on the Y-Z plane.
- **Spectral Analysis**: Normalized PSD of magnetic field fluctuations using Welch's method (simplified).
- **Kp Index Estimator**: Rough estimation of geomagnetic activity based on IMF Bz polarity and magnitude.

---

## 🤝 Attribution & References

- **Live Data Source**: [NOAA SWPC DSCOVR](https://services.swpc.noaa.gov/)
- **Mission Reference**: [ISRO Aditya-L1 Mission](https://www.isro.gov.in/Aditya_L1.html)
- **Analysis Methodology**: *Analysis Methods for Multi-Spacecraft Data*, Paschmann & Daly, ISSI SR-001.

---

## 🛠️ Development & Support

Developed by **Vivek Shrivastav**. Project focus: High-fidelity scientific communication and real-time space weather telemetry.
