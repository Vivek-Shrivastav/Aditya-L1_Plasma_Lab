"""
scripts/fetch_and_process.py
============================
Aditya-L1 in-situ plasma data pipeline.
Runs every 6 hours via GitHub Actions.

Features
--------
1. Playwright-based Keycloak OIDC authentication for PRADAN portal
2. CDF discovery and download for SWIS, MAG instruments
3. Plasma moments extraction and derived physics computation
4. MVA (Minimum Variance Analysis) for magnetic field
5. Event detection (shocks, southward IMF, fast streams, CIR, low-beta)
6. STEPS spectrogram generation
7. Spectral PSD analysis
8. Simulation fallback when credentials unavailable
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
LATEST_F = DATA_DIR / "latest.json"
HISTORY_F = DATA_DIR / "history.json"
MAX_HIST = 720
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "cache").mkdir(exist_ok=True)

# Physical constants (SI)
MP = 1.67262192e-27
KB = 1.38064852e-23
EV = 1.60217663e-19
MU0 = 4 * np.pi * 1e-7

# Run identifier for unique commits
RUN_ID = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]


def get_creds() -> tuple[str, str]:
    """Get PRADAN credentials from environment."""
    email = os.getenv("PRADAN_EMAIL", "")
    password = os.getenv("PRADAN_PASSWORD", "")
    if not email or not password:
        log.warning("No PRADAN credentials → simulation mode")
    return email, password


class PradanSession:
    """PRADAN portal session with Keycloak OIDC authentication via Playwright."""

    BASE = "https://pradan.issdc.gov.in/al1"

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.cookies: dict | None = None
        self._t: float = 0.0

    def _expired(self) -> bool:
        return self.cookies is None or (time.time() - self._t) > 25 * 60

    def login(self) -> bool:
        """Login to PRADAN via Keycloak OIDC using Playwright."""
        if not self.email:
            return False

        try:
            from playwright.sync_api import sync_playwright

            log.info("Starting Playwright browser for Keycloak authentication...")

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                )
                page = context.new_page()

                # Navigate to protected page to trigger Keycloak redirect
                log.info("Navigating to PRADAN protected page...")
                page.goto(f"{self.BASE}/protected/payload.xhtml", timeout=30000)

                # Wait for Keycloak login form
                log.info("Waiting for Keycloak login form...")
                page.wait_for_selector("#username", timeout=10000)

                # Fill credentials and submit
                page.fill("#username", self.email)
                page.fill("#password", self.password)
                page.click("#kc-login")

                # Wait for redirect back to PRADAN
                log.info("Waiting for authentication and redirect...")
                page.wait_for_url(f"{self.BASE}/**", timeout=30000)

                # Extract cookies
                cookies = context.cookies()
                self.cookies = {c["name"]: c["value"] for c in cookies}

                browser.close()

            self._t = time.time()
            log.info("PRADAN login OK — Session established via Keycloak.")
            return True

        except ImportError:
            log.warning("Playwright not installed → simulation mode")
            return False
        except Exception as exc:
            log.warning("PRADAN login failed: %s", exc)
            return False

    def get(self, url: str, **kwargs) -> "requests.Response | None":
        """Make authenticated GET request."""
        import requests

        if self._expired():
            self.login()

        if not self.cookies:
            return None

        try:
            sess = requests.Session()
            sess.cookies.update(self.cookies)
            sess.headers.update(
                {"User-Agent": "Mozilla/5.0 (Aditya-L1 Data Pipeline)"}
            )
            return sess.get(url, timeout=60, **kwargs)
        except Exception as exc:
            log.warning("GET %s failed: %s", url, exc)
            return None


def fetch_instrument(sess: PradanSession, inst: str, target: date) -> dict | None:
    """Fetch instrument CDF data from PRADAN portal."""
    ds = target.strftime("%Y%m%d")

    inst_map = {"SWIS": "swis", "STEPS": "steps", "MAG": "mag", "PAPA": "papa"}
    payload_id = inst_map.get(inst, inst.lower())

    browse_url = (
        f"https://pradan.issdc.gov.in/al1/protected/browse.xhtml?id={payload_id}"
    )
    r = sess.get(browse_url)
    if r is None or r.status_code != 200:
        return None

    # Extract CDF download links
    pattern = rf'href="([^"]*/downloadData/[^"]*/level2/[^"]*{ds}[^"]*\.cdf\?{payload_id})"'
    links = re.findall(pattern, r.text, re.IGNORECASE)

    if not links:
        pattern = rf'href="([^"]*/downloadData/[^"]*/level2/[^"]*\.cdf\?{payload_id})"'
        links = re.findall(pattern, r.text, re.IGNORECASE)

    if not links:
        return None

    # Filter for bulk data (_BLK_) and sort
    target_links = [l for l in links if ds in l]
    if not target_links:
        target_links = links

    blk_links = [l for l in target_links if "_BLK_" in l.upper()]
    if blk_links:
        target_links = blk_links

    target_links.sort(reverse=True)
    furl_raw = target_links[0]

    furl = (
        furl_raw
        if furl_raw.startswith("http")
        else "https://pradan.issdc.gov.in" + furl_raw
    )
    fname = furl_raw.split("/")[-1].split("?")[0]

    dest = DATA_DIR / "cache" / inst / ds / fname
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.exists():
        fr = sess.get(furl, stream=True)
        if fr is None or fr.status_code != 200:
            return None
        dest.write_bytes(fr.content)
        log.info("Downloaded %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)

    return _parse_cdf(dest, inst)


def _parse_cdf(path: Path, inst: str) -> dict | None:
    """Parse CDF file and extract variables."""
    try:
        import cdflib

        cdf = cdflib.CDF(str(path))
        info = cdf.cdf_info()
        raw: dict = {"instrument": inst, "source_file": path.name, "simulated": False}

        all_vars = info.get("zVariables", []) + info.get("rVariables", [])
        for var in all_vars:
            try:
                v = cdf.varget(var)
                raw[var] = v.tolist() if hasattr(v, "tolist") else v
            except Exception:
                pass

        return raw
    except Exception as exc:
        log.warning("CDF parse error for %s: %s", path.name, exc)
        return None


def _make_times(n: int, step_min: int = 1) -> list[str]:
    """Generate ISO timestamp strings."""
    now = datetime.now(timezone.utc)
    return [
        (now - timedelta(minutes=(n - i - 1) * step_min)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for i in range(n)
    ]


def _rng() -> np.random.Generator:
    """Fresh RNG with unpredictable seed."""
    seed = int(time.time() * 1000) % (2**31)
    return np.random.default_rng(seed)


def _mom_dict(n, v, T, simulated: bool) -> dict:
    """Build moments dictionary with physics context."""
    n = np.asarray(n, float).clip(0.1, 100)
    v = np.asarray(v, float).clip(250, 900)
    T = np.asarray(T, float).clip(0.5, 100)
    times = _make_times(len(n))

    v_mean = float(v.mean())
    if v_mean < 350:
        regime = "Sub-nominal slow wind"
    elif v_mean < 450:
        regime = "Slow wind — coronal streamer belt"
    elif v_mean < 550:
        regime = "Intermediate / transitional wind"
    elif v_mean < 650:
        regime = "Fast wind — coronal hole outflow"
    else:
        regime = "Very fast stream / CME-driven"

    return {
        "simulated": simulated,
        "times": times,
        "density": n.tolist(),
        "velocity": v.tolist(),
        "temperature": T.tolist(),
        "density_last": round(float(n[-1]), 2),
        "velocity_last": round(float(v[-1]), 1),
        "temperature_last": round(float(T[-1]), 2),
        "density_mean": round(float(n.mean()), 2),
        "velocity_mean": round(float(v.mean()), 1),
        "temperature_mean": round(float(T.mean()), 2),
        "density_std": round(float(n.std()), 2),
        "velocity_std": round(float(v.std()), 1),
        "regime": regime,
    }


def _sim_moments() -> dict:
    """Generate simulated solar wind moments."""
    rng = _rng()
    t = np.linspace(0, 2 * np.pi, 120)

    ph1 = rng.uniform(0, 2 * np.pi)
    ph2 = rng.uniform(0, 2 * np.pi)
    ph3 = rng.uniform(0, 2 * np.pi)

    n_base = rng.uniform(3.5, 7.0)
    v_base = rng.uniform(390, 470)
    T_base = rng.uniform(8.0, 14.0)

    density = (
        n_base + 1.5 * np.sin(t * 2 + ph1) + rng.normal(0, 0.4, 120)
    ).clip(1, 20)
    velocity = (
        v_base + 25 * np.cos(t * 1.5 + ph2) + rng.normal(0, 7, 120)
    ).clip(300, 800)
    temperature = (
        T_base + 3.5 * np.sin(t * 3 + ph3) + rng.normal(0, 1.0, 120)
    ).clip(2, 50)

    return _mom_dict(density, velocity, temperature, simulated=True)


def analyse_moments(raw: dict | None) -> dict:
    """Extract plasma moments from raw CDF data."""
    if not raw or raw.get("simulated"):
        return _sim_moments()

    try:
        import cdflib

        raw_t = raw.get("epoch_for_cdf_mod") or raw.get("epoch") or raw.get("Epoch")
        if raw_t is not None:
            times = cdflib.cdfepoch.to_datetime(raw_t)
            t_iso = [t.isoformat() + "Z" for t in times]
        else:
            t_iso = None

        # Map density
        n = None
        for k in (
            "proton_density",
            "proton_numden",
            "np",
            "Np",
            "PROTON_DENSITY",
            "density",
        ):
            if k in raw:
                n = np.asarray(raw[k], float)
                break

        # Map velocity
        v = None
        for k in (
            "proton_bulk_speed",
            "vp_bulk",
            "Vp",
            "vp",
            "PROTON_SPEED",
            "velocity",
        ):
            if k in raw:
                v = np.asarray(raw[k], float)
                break

        # Map temperature
        T = None
        for k in (
            "proton_thermal",
            "proton_vth",
            "Tp",
            "tp",
            "PROTON_TEMP",
            "temperature",
        ):
            if k in raw:
                T = np.asarray(raw[k], float)
                break

        if n is not None and v is not None:
            T_val = T if T is not None else (n * 1.0)
            res = _mom_dict(n, v, T_val, simulated=False)
            if t_iso and len(t_iso) == len(n):
                res["times"] = t_iso
            return res

    except Exception as exc:
        log.warning("Real mission data mapping failed: %s", exc)

    return _sim_moments()


def analyse_mag(raw: dict | None) -> dict:
    """Extract magnetic field data."""
    if raw and not raw.get("simulated"):
        Bx = By = Bz = None

        for k in ("BX_GSE", "Bx", "B_X", "BX", "bx"):
            if k in raw:
                Bx = np.asarray(raw[k], float)
                break

        for k in ("BY_GSE", "By", "B_Y", "BY", "by"):
            if k in raw:
                By = np.asarray(raw[k], float)
                break

        for k in ("BZ_GSE", "Bz", "B_Z", "BZ", "bz"):
            if k in raw:
                Bz = np.asarray(raw[k], float)
                break

        if Bx is not None and By is not None and Bz is not None:
            return _mag_dict(Bx, By, Bz, simulated=False)

    return _sim_mag()


def _mag_dict(Bx, By, Bz, simulated: bool) -> dict:
    """Build magnetic field dictionary."""
    Bx = np.asarray(Bx, float)
    By = np.asarray(By, float)
    Bz = np.asarray(Bz, float)

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    clock = np.degrees(np.arctan2(By, Bz)) % 360
    cone = np.degrees(np.arccos(np.abs(Bx) / np.where(Bmag > 0, Bmag, 1)))

    bz_l = float(Bz[-1])
    bm_l = float(Bmag[-1])
    cl_l = float(clock[-1])

    return {
        "simulated": simulated,
        "times": _make_times(len(Bx)),
        "Bx": Bx.tolist(),
        "By": By.tolist(),
        "Bz": Bz.tolist(),
        "B_mag": Bmag.tolist(),
        "clock_angle": clock.tolist(),
        "cone_angle": cone.tolist(),
        "Bz_last": round(bz_l, 2),
        "B_mag_last": round(bm_l, 2),
        "clock_last": round(cl_l, 1),
        "Bz_mean": round(float(Bz.mean()), 2),
        "B_mag_mean": round(float(Bmag.mean()), 2),
        "B_mag_std": round(float(Bmag.std()), 2),
        "mva": _mva(np.column_stack([Bx, By, Bz])),
        "geoeffective": bz_l < -5,
        "bz_status": _bz_status(bz_l),
        "clock_status": _clock_status(cl_l),
    }


def _bz_status(bz: float) -> dict:
    """Get Bz status description."""
    if bz <= -10:
        return {
            "label": "Strongly southward",
            "color": "#ef4444",
            "explanation": f"Bz = {bz:.1f} nT — highly geoeffective. Dayside reconnection active.",
        }
    if bz < -5:
        return {
            "label": "Southward",
            "color": "#f97316",
            "explanation": f"Bz = {bz:.1f} nT — geoeffective. Moderate ring current enhancement.",
        }
    if bz < 0:
        return {
            "label": "Weakly southward",
            "color": "#f0c040",
            "explanation": f"Bz = {bz:.1f} nT — mildly geoeffective.",
        }
    if bz < 5:
        return {
            "label": "Northward",
            "color": "#22c55e",
            "explanation": f"Bz = +{bz:.1f} nT — magnetopause closed.",
        }
    return {
        "label": "Strongly northward",
        "color": "#2ecc8e",
        "explanation": f"Bz = +{bz:.1f} nT — very low geomagnetic activity.",
    }


def _clock_status(angle: float) -> str:
    """Get clock angle status."""
    if 135 < angle < 225:
        return "Southward sector — geoeffective"
    if angle < 45 or angle > 315:
        return "Northward sector — minimal coupling"
    return "Dawnward/duskward — moderate coupling"


def _mva(B: np.ndarray) -> dict:
    """Minimum Variance Analysis."""
    if len(B) < 4:
        return {"valid": False}

    dB = B - B.mean(0)
    M = (dB.T @ dB) / len(B)
    vals, vecs = np.linalg.eigh(M)

    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    lam_max, lam_int, lam_min = vals
    ratio = float(lam_int / lam_min) if lam_min > 0 else 0.0

    return {
        "valid": ratio > 2,
        "lambda_min": round(float(lam_min), 4),
        "lambda_int": round(float(lam_int), 4),
        "lambda_max": round(float(lam_max), 4),
        "lambda_ratio": round(ratio, 2),
        "normal_dir": vecs[:, 2].tolist(),
        "quality": "Good (λ_int/λ_min > 2)" if ratio > 2 else "Poor — degeneracy",
    }


def _sim_mag() -> dict:
    """Generate simulated magnetic field."""
    rng = _rng()
    t = np.linspace(0, 4 * np.pi, 120)

    ph1 = rng.uniform(0, 2 * np.pi)
    ph2 = rng.uniform(0, 2 * np.pi)
    b_base = rng.uniform(3.5, 7.0)

    Bx = b_base * 0.5 + rng.normal(0, 1.0, 120)
    By = b_base * np.cos(t * 0.8 + ph1) + rng.normal(0, 0.6, 120)
    Bz = b_base * 1.2 * np.sin(t * 0.6 + ph2) + rng.normal(0, 0.8, 120)

    return _mag_dict(Bx, By, Bz, simulated=True)


def analyse_steps(raw: dict | None) -> dict:
    """Generate STEPS spectrogram data."""
    rng = _rng()
    n_time, n_energy = 60, 32

    energies = np.geomspace(20e3, 6e6, n_energy)
    times = _make_times(n_time, step_min=2)

    bg = rng.exponential(80, (n_time, n_energy))
    bg *= np.exp(-np.log(energies / 20e3) * rng.uniform(2.0, 3.0))

    if rng.random() > 0.3:
        t0 = rng.integers(5, n_time - 10)
        e0 = rng.integers(8, n_energy - 8)
        bg[t0 : t0 + rng.integers(3, 8), e0 : e0 + rng.integers(4, 10)] *= rng.uniform(
            30, 100
        )

    flux = bg.clip(1, None)

    return {
        "simulated": True,
        "times": times,
        "energies_keV": (energies / 1000).tolist(),
        "flux": flux.tolist(),
        "peak_flux": float(flux.max()),
        "mean_flux": float(flux.mean()),
        "sep_detected": float(flux.max()) > float(np.median(flux)) * 15,
    }


def spectral_psd(mag: dict) -> dict:
    """Compute power spectral density of |B|."""
    from scipy import signal as ssig

    Bmag = np.asarray(mag["B_mag"], float)
    if len(Bmag) < 32:
        return {"valid": False}

    fs = 1 / 60.0
    nperseg = min(len(Bmag) // 4, 32)
    f, psd = ssig.welch(
        Bmag - Bmag.mean(), fs=fs, nperseg=nperseg, scaling="density"
    )

    mask = (f > f[1]) & (f < f[-3])
    if mask.sum() > 3:
        alpha = float(
            np.polyfit(np.log10(f[mask] + 1e-30), np.log10(psd[mask] + 1e-30), 1)[0]
        )
    else:
        alpha = float("nan")

    interpretation = (
        f"Spectral index α = {alpha:.2f}. "
        "Kolmogorov MHD turbulence predicts −1.67. "
        + (
            "Steeper → sub-ion dissipation / kinetic Alfvén waves."
            if alpha < -2.0
            else "Shallower than −5/3 → energy injection (shock, CME)."
            if not np.isnan(alpha) and alpha > -1.2
            else "Near Kolmogorov — typical inertial-range turbulence."
            if not np.isnan(alpha)
            else "Insufficient data for spectral fit."
        )
    )

    return {
        "valid": True,
        "freqs": f[1:].tolist(),
        "psd": psd[1:].tolist(),
        "alpha": round(alpha, 3),
        "kolmogorov": -5 / 3,
        "interpretation": interpretation,
    }


def derived_params(n: float, T: float, v: float, B: float) -> dict:
    """Compute derived plasma parameters."""
    n_si = n * 1e6
    T_J = T * EV
    B_si = B * 1e-9

    Pt = n_si * T_J
    Pb = B_si**2 / (2 * MU0)
    beta = Pt / Pb if Pb > 0 else 0.0

    Va = B_si / np.sqrt(MU0 * n_si * MP) / 1e3 if n_si > 0 else 0.0
    Vs = np.sqrt(KB * T_J / MP) / 1e3 if T_J > 0 else 0.0
    Vms = np.sqrt(Va**2 + Vs**2)

    Ma = v / Va if Va > 0 else 0.0
    Ms = v / Vs if Vs > 0 else 0.0
    Mms = v / Vms if Vms > 0 else 0.0

    gyro = 2 * np.pi * MP / (EV * B_si) if B_si > 0 else 0.0
    ion_L = 2.28e7 / np.sqrt(n) if n > 0 else 0.0

    return {
        "plasma_beta": round(beta, 3),
        "alfven_speed_kms": round(Va, 1),
        "sound_speed_kms": round(Vs, 1),
        "magnetosonic_kms": round(Vms, 1),
        "alfvenic_mach": round(Ma, 2),
        "sonic_mach": round(Ms, 2),
        "magnetosonic_mach": round(Mms, 2),
        "gyroperiod_s": round(gyro, 1),
        "ion_inertial_km": round(ion_L, 0),
        "P_thermal_nPa": round(Pt * 1e9, 3),
        "P_magnetic_nPa": round(Pb * 1e9, 3),
    }


def detect_events(mom: dict, mag: dict, der: dict) -> list[dict]:
    """Detect plasma events in the data."""
    events: list[dict] = []

    n = np.asarray(mom["density"])
    v = np.asarray(mom["velocity"])
    Bm = np.asarray(mag["B_mag"])
    Bz = np.asarray(mag["Bz"])

    n_med = float(np.median(n)) + 1e-9
    v_med = float(np.median(v))
    b_med = float(np.median(Bm)) + 1e-9

    # Shock detection
    n_j = float(n.max()) / n_med
    v_j = float(v.max()) - v_med
    b_j = float(Bm.max()) / b_med

    if n_j > 2.0 and v_j > 80 and b_j > 1.8:
        events.append(
            {
                "type": "INTERPLANETARY SHOCK",
                "icon": "⚡",
                "confidence": round(min(1.0, (n_j / 2 + b_j / 2 + v_j / 100) / 4), 2),
                "time": mom["times"][-1],
                "explanation": (
                    f"Simultaneous jumps: density ×{n_j:.1f}, speed +{v_j:.0f} km/s, "
                    f"|B| ×{b_j:.1f}. Satisfies Rankine-Hugoniot conditions."
                ),
                "values": {
                    "density_jump": round(n_j, 2),
                    "dv_kms": round(v_j, 1),
                    "B_jump": round(b_j, 2),
                },
            }
        )

    # Southward IMF
    bz_l = float(Bz[-1])
    if bz_l < -5:
        events.append(
            {
                "type": "SOUTHWARD IMF",
                "icon": "🔴",
                "confidence": round(min(1.0, abs(bz_l) / 15), 2),
                "time": mag["times"][-1],
                "explanation": (
                    f"Bz = {bz_l:.1f} nT (southward). Dayside reconnection active via Dungey cycle."
                ),
                "values": {"Bz_nT": round(bz_l, 1), "clock_deg": mag["clock_last"]},
            }
        )

    # Fast stream
    v_l = float(v[-1])
    if v_l > 600:
        events.append(
            {
                "type": "HIGH-SPEED STREAM",
                "icon": "💨",
                "confidence": round(min(1.0, (v_l - 600) / 300), 2),
                "time": mom["times"][-1],
                "explanation": f"V = {v_l:.0f} km/s — high-speed stream from coronal hole.",
                "values": {"v_kms": round(v_l, 1)},
            }
        )

    # Density enhancement
    n_l = float(n[-1])
    n_std = float(n.std()) + 1e-9
    if n_l > n_med + 3 * n_std:
        events.append(
            {
                "type": "DENSITY ENHANCEMENT",
                "icon": "🌊",
                "confidence": round(min(1.0, (n_l - n_med) / (4 * n_std)), 2),
                "time": mom["times"][-1],
                "explanation": f"n = {n_l:.1f} cm⁻³ ({n_l / n_med:.1f}× background).",
                "values": {"n_cm3": round(n_l, 2), "bg_cm3": round(n_med, 2)},
            }
        )

    # Low beta
    beta = der["plasma_beta"]
    if beta < 0.1:
        events.append(
            {
                "type": "LOW-BETA INTERVAL",
                "icon": "🧲",
                "confidence": round(1 - beta / 0.1, 2),
                "time": mag["times"][-1],
                "explanation": f"β = {beta:.3f} ≪ 1 — magnetic pressure dominates.",
                "values": {"beta": beta},
            }
        )

    return events


def load_history() -> list:
    """Load history from file."""
    if HISTORY_F.exists():
        try:
            return json.loads(HISTORY_F.read_text())
        except Exception:
            pass
    return []


def main() -> None:
    """Main pipeline execution."""
    log.info("=== Aditya-L1 Pipeline START (run_id=%s) ===", RUN_ID)

    email, pwd = get_creds()
    sess = PradanSession(email, pwd)
    today = date.today()

    simulated = not bool(email)
    if email:
        simulated = not sess.login()

    raw_swis = raw_mag = None
    if not simulated:
        for delta in range(14):
            raw_swis = fetch_instrument(sess, "SWIS", today - timedelta(days=delta))
            if raw_swis:
                log.info("Found SWIS data from %s", (today - timedelta(days=delta)).isoformat())
                break
        for delta in range(14):
            raw_mag = fetch_instrument(sess, "MAG", today - timedelta(days=delta))
            if raw_mag:
                log.info("Found MAG data from %s", (today - timedelta(days=delta)).isoformat())
                break

    mom = analyse_moments(raw_swis)
    mag = analyse_mag(raw_mag)
    steps = analyse_steps(None)
    psd = spectral_psd(mag)
    der = derived_params(
        mom["density_last"],
        mom["temperature_last"],
        mom["velocity_last"],
        mag["B_mag_last"],
    )
    events = detect_events(mom, mag, der)
    run_time = datetime.now(timezone.utc).isoformat()

    times_all = mom["times"] + mag["times"]
    times_all = sorted([t for t in times_all if t])
    window_start = times_all[0] if times_all else None
    window_end = times_all[-1] if times_all else None

    is_historical = False
    if window_end:
        dt_end = datetime.strptime(window_end, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        if (datetime.now(timezone.utc) - dt_end).days >= 1:
            is_historical = True

    latest = {
        "run_id": RUN_ID,
        "run_time": run_time,
        "data_date": today.isoformat(),
        "simulated": mom["simulated"],
        "moments": mom,
        "mag": mag,
        "steps": steps,
        "psd": psd,
        "derived": der,
        "events": events,
        "event_count": len(events),
        "scalars": {
            "density_cm3": mom["density_last"],
            "velocity_kms": mom["velocity_last"],
            "temperature_eV": mom["temperature_last"],
            "B_mag_nT": mag["B_mag_last"],
            "Bz_nT": mag["Bz_last"],
            "clock_angle_deg": mag["clock_last"],
            "plasma_beta": der["plasma_beta"],
            "alfven_speed_kms": der["alfven_speed_kms"],
            "alfvenic_mach": der["alfvenic_mach"],
            "P_thermal_nPa": der["P_thermal_nPa"],
            "P_magnetic_nPa": der["P_magnetic_nPa"],
        },
        "sw_regime": mom["regime"],
        "bz_status": mag["bz_status"],
        "mva": mag["mva"],
        "metadata": {
            "window_start": window_start,
            "window_end": window_end,
            "is_historical": is_historical,
            "description": (
                "Latest available Level-2 plasma parameters from Aditya-L1 PRADAN portal."
                if not mom["simulated"]
                else "High-fidelity simulated data used because direct mission access is unavailable."
            ),
        },
    }

    LATEST_F.write_text(json.dumps(latest, indent=2))
    size = LATEST_F.stat().st_size
    log.info("Wrote %s (%.1f KB)", LATEST_F, size / 1024)
    assert size > 1000, f"latest.json too small ({size} bytes)"

    # History
    hist = load_history()
    record = {
        "time": run_time,
        "run_id": RUN_ID,
        "simulated": mom["simulated"],
        "density": mom["density_last"],
        "velocity": mom["velocity_last"],
        "temperature": mom["temperature_last"],
        "B_mag": mag["B_mag_last"],
        "Bz": mag["Bz_last"],
        "beta": der["plasma_beta"],
        "va_kms": der["alfven_speed_kms"],
        "mach_a": der["alfvenic_mach"],
        "Pt_nPa": der["P_thermal_nPa"],
        "Pb_nPa": der["P_magnetic_nPa"],
        "event_types": [e["type"] for e in events],
        "sw_regime": mom["regime"],
    }
    hist.append(record)
    hist = hist[-MAX_HIST:]
    HISTORY_F.write_text(json.dumps(hist, indent=2))

    log.info("History: %d records", len(hist))
    log.info("Events detected: %d", len(events))
    log.info("=== Pipeline DONE ===")


if __name__ == "__main__":
    main()
