# ☀️ Aditya-L1 Plasma Lab

> A real-time solar wind monitoring dashboard for India's first dedicated solar mission — [Aditya-L1](https://www.isro.gov.in/Aditya_L1.html) — orbiting the Sun–Earth L1 Lagrange point.

🔗 **Live site**: https://vivek-shrivastav.github.io/Aditya-L1_Plasma_Lab/

---

## What does this show?

The dashboard streams live measurements of the **solar wind** — a continuous plasma flow from the Sun carrying the Interplanetary Magnetic Field (IMF). These measurements give **30–60 minutes advance warning** of space weather events that can affect satellites, power grids, and GPS systems.

| Parameter | Unit | Why it matters |
|---|---|---|
| Proton Density | cm⁻³ | Elevated (> 10) signals compressed plasma regions |
| Bulk Speed | km/s | > 500 km/s = high-speed stream; > 700 = extreme |
| Proton Temperature | eV | CME heating pushes this above 50 eV |
| IMF Bz | nT | Southward (< −5 nT) drives geomagnetic storms |
| Plasma β | — | Ratio of thermal to magnetic pressure |
| Alfvén Speed | km/s | Magnetic wave propagation speed |
| IMF Clock Angle | ° | 0° = quiet; 180° = southward/active |

---

## Architecture

```
Aditya-L1_Plasma_Lab/
├── index.html            # Single-page application shell
├── css/style.css         # Styles (glassmorphism, responsive, tooltips)
├── js/app.js             # SPA engine — data fetching, charting, navigation
├── scripts/
│   └── fetch_and_process.py  # Python pipeline (runs via GitHub Actions)
├── data/
│   ├── latest.json       # Most recent pipeline run output (auto-updated)
│   └── history.json      # Rolling history of pipeline runs (up to 720)
├── assets/               # Static images
├── .env.example          # Environment variable template
├── pyproject.toml        # Python linting/formatting config (ruff + black)
└── requirements-workflow.txt  # Pipeline Python dependencies
```

### Data flow

```
GitHub Actions (every 6 h)
  └── scripts/fetch_and_process.py
        ├── Authenticates to PRADAN portal (or uses simulation)
        ├── Downloads SWIS (plasma) + MAG CDF files
        ├── Computes derived quantities (β, VA, MVA, PSD, events)
        └── Writes data/latest.json + data/history.json → git push

Browser (every 60 s)
  ├── fetch data/latest.json   ← rich Aditya-L1 scalars (primary)
  ├── fetch data/history.json  ← historical trend charts
  └── fetch NOAA SWPC DSCOVR   ← high-cadence 7-day time-series (fallback/supplement)
```

### Navigation

| Page | Content |
|---|---|
| **Home** | Live snapshot of all key parameters with units, badges, and tooltips |
| **Dashboard** | Interactive 7-day time-series charts (Plotly) |
| **History** | Aditya-L1 pipeline historical trends (density, speed, Bz, β) |
| **Visualizer** | IMF clock angle dial, Kp estimator, particle animation |
| **Explorer** | Payload specs, space-weather primer, glossary |
| **Analysis Lab** | PSD spectral analysis, Alfvén speed calculator, event log |

---

## Local development

The frontend is a plain static site — no build step required.

### Run locally

```bash
# Python 3 built-in server
python3 -m http.server 8080 --directory .
# Then open: http://localhost:8080
```

> **Note**: Fetching `data/latest.json` requires an HTTP server (not `file://`).  
> The NOAA SWPC live feed works from any origin.

### Run the Python pipeline

```bash
# Install dependencies
pip install -r requirements-workflow.txt
# (Optional) Install Playwright browser for PRADAN authentication
playwright install chromium

# Copy and fill in credentials
cp .env.example .env
# Edit .env — leave blank to run in simulation mode

# Run pipeline
python scripts/fetch_and_process.py
```

Output files: `data/latest.json` and `data/history.json`.

---

## Configuration

| Variable | Description |
|---|---|
| `PRADAN_EMAIL` | Your PRADAN portal email (leave blank for simulation) |
| `PRADAN_PASSWORD` | Your PRADAN portal password |

Set these as **GitHub repository secrets** (`Settings → Secrets → Actions`) for the automated pipeline.

---

## Python development tools

```bash
pip install ruff black

# Lint
ruff check scripts/

# Format
black scripts/
```

---

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JavaScript (SPA), HTML5, CSS3 |
| Charts | [Plotly.js](https://plotly.com/javascript/) |
| Fonts | Inter, Space Grotesk, JetBrains Mono (Google Fonts) |
| Backend pipeline | Python 3.11 · numpy · scipy · cdflib · Playwright |
| Deployment | GitHub Pages (static) |
| Data pipeline CI | GitHub Actions (every 6 h + manual trigger) |

---

## Data sources

- **Aditya-L1 / PRADAN**: https://pradan.issdc.gov.in/ — official ISRO data portal for Aditya-L1 instrument data
- **NOAA SWPC DSCOVR**: https://services.swpc.noaa.gov/ — high-cadence (~1 min) 7-day solar wind time-series from DSCOVR at L1

---

## References

- ISRO Aditya-L1 Mission: https://www.isro.gov.in/Aditya_L1.html
- NOAA Space Weather Prediction Center: https://www.swpc.noaa.gov/
- Analysis methodology: *Analysis Methods for Multi-Spacecraft Data*, Paschmann & Daly, ISSI SR-001

---

Developed by **Vivek Shrivastav**. Contributions welcome — see [issues](../../issues).
