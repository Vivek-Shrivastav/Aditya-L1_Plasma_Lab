# 🌞 Aditya-L1 Plasma Observatory

> One-stop repository for ISRO's Aditya-L1 in-situ plasma science — from mission context and instrument tutorials to live data, interactive visualization, and analysis tools grounded in the ISSI SR-001 methodology.

🔗 **Live site**: https://vivek-shrivastav.github.io/Aditya-L1_Plasma_observatory/

---

## Site Structure

| Page | Purpose |
|------|---------|
| [`index.html`](index.html) | **Learn first** — mission context, instruments, data tutorials, analysis methods |
| [`dashboard.html`](dashboard.html) | **Live monitor** — real-time panels, auto-updated every 6h |
| [`visualizer.html`](visualizer.html) | **Data explorer** — pick any date/time, upload CDF, instant plots |
| [`lab.html`](lab.html) | **Analysis lab** — spectral, MVA, shock tools (see branch `lab`) |
| [`explorer.html`](explorer.html) | **Science reference** — full interactive instrument explorer |

---

## How It Works

```
GitHub Actions (every 6 hours)
  └── scripts/fetch_and_process.py
        ├── Login to PRADAN (PRADAN_EMAIL + PRADAN_PASSWORD secrets)
        ├── Download latest Level-2 CDF files
        ├── Run analysis:
        │     ├── Plasma moments (n, V, T) from SWIS/PAPA
        │     ├── MAG vector field analysis (MVA, clock angle, PSD)
        │     ├── STEPS spectrogram
        │     ├── Derived quantities (β, V_A, Mach numbers)
        │     └── Event detection (shocks, southward IMF, fast streams)
        ├── Write data/latest.json   ← all pages read this
        └── Write data/history.json  ← 6-month rolling record

GitHub Pages serves all HTML files statically.
No server, no backend — just files + scheduled Actions.
```

---

## Setup (5 min)

**1. Enable GitHub Pages**
Settings → Pages → Source: Deploy from branch → main → / (root) → Save

**2. Add PRADAN credentials as Secrets**
Settings → Secrets and variables → Actions:
- `PRADAN_EMAIL` → your PRADAN email
- `PRADAN_PASSWORD` → your PRADAN password

> Register free at [pradan.issdc.gov.in](https://pradan.issdc.gov.in)

**3. Run the first workflow manually**
Actions → Fetch & Analyse Aditya-L1 Data → Run workflow

**4. Open your live site**
`https://YOUR_USERNAME.github.io/REPO_NAME/`

---

## Without Credentials

The pipeline runs fine without credentials — it generates **physically realistic simulated solar wind data** (clearly labelled) so all visualizations, event detection, and analysis tools work immediately for demonstration.

---

## Analysis Methods

Methods follow **Analysis Methods for Multi-Spacecraft Data** (Paschmann & Daly, ISSI SR-001):

| Method | Chapter | Tool |
|--------|---------|------|
| Spectral analysis (FFT/Welch/Morlet) | Ch. 1 | `lab.html#spectral` |
| Plasma moments (n, V, T, tensor) | Ch. 5–6 | `lab.html#moments` |
| Minimum Variance Analysis (MVA) | Ch. 8 | `lab.html#mva` + dashboard |
| Rankine-Hugoniot shocks | Ch. 10 | `lab.html#shock` |
| Time series resampling/filtering | Ch. 2 | `lab.html#timeseries` |
| Velocity distributions + anisotropy | Ch. 5 | `lab.html#anisotropy` |

---

## New in This Version

- **Learn-first flow**: `index.html` teaches mission, instruments, physics, and methods before directing users to experiment
- **Per-plot timestamps**: every chart shows exact UTC data window, source, and instrument
- **Interactive data visualizer**: pick any date/time or upload a CDF file
- **Tooltips on every metric card**: hover for physical interpretation
- **Richer event detection**: shocks (R-H conditions), southward IMF (Dungey cycle), fast streams, density enhancements, low-beta intervals
- **MVA table**: eigenvalues, normal direction, quality flag on the dashboard

---

## Attribution

- Mission data © **ISRO / ISSDC** · Data via [pradan.issdc.gov.in](https://pradan.issdc.gov.in)
- Analysis methodology: *Analysis Methods for Multi-Spacecraft Data*, Paschmann & Daly, ISSI SR-001
- Reference: [Vivek-Shrivastav/Two\_Stream\_Lab](https://github.com/Vivek-Shrivastav/Two_Stream_Lab) (architecture pattern)
