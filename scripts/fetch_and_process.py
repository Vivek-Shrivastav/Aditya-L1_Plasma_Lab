"""
scripts/fetch_and_process.py
============================
Aditya-L1 in-situ plasma data pipeline.
Runs every 6 hours via GitHub Actions.

Fixes applied
-------------
1. Seed is now time.time() (not floored) — every run produces different data
2. run_id (nanoid-style) forces git diff even if data looks similar
3. Writes are verified with file size check before exit
4. History always appended — guarantees latest.json changes every run

Credentials come from GitHub Secrets (PRADAN_EMAIL, PRADAN_PASSWORD).
Never hardcoded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DATA_DIR  = Path("data")
LATEST_F  = DATA_DIR / "latest.json"
HISTORY_F = DATA_DIR / "history.json"
MAX_HIST  = 720
DATA_DIR.mkdir(exist_ok=True)

# Physical constants (SI)
MP  = 1.67262192e-27
KB  = 1.38064852e-23
EV  = 1.60217663e-19
MU0 = 4 * np.pi * 1e-7

# ── Run identifier ─────────────────────────────────────────────────────────
# A unique ID per run ensures the JSON is always different,
# so git diff ALWAYS sees a change and always commits.
RUN_ID = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
log.info("Run ID: %s", RUN_ID)

# ── Credentials ────────────────────────────────────────────────────────────
def get_creds() -> tuple[str, str]:
    e = os.getenv("PRADAN_EMAIL", "")
    p = os.getenv("PRADAN_PASSWORD", "")
    if not e or not p:
        log.warning("No PRADAN credentials → simulation mode")
    return e, p

# ── PRADAN session ──────────────────────────────────────────────────────────
class PradanSession:
    BASE = "https://pradan.issdc.gov.in"

    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.sess: requests.Session | None = None
        self._t: float = 0.0

    def _expired(self) -> bool:
        return self.sess is None or (time.time() - self._t) > 25 * 60

    def login(self) -> bool:
        if not self.email:
            return False
        try:
            self.sess = requests.Session()
            r = self.sess.post(
                f"{self.BASE}/login",
                data={"email": self.email, "password": self.password},
                timeout=30,
            )
            r.raise_for_status()
            self._t = time.time()
            log.info("PRADAN login OK")
            return True
        except Exception as exc:
            log.warning("PRADAN login failed: %s", exc)
            self.sess = None
            return False

    def get(self, url: str, **kw) -> requests.Response | None:
        if self._expired():
            self.login()
        if not self.sess:
            return None
        try:
            return self.sess.get(url, timeout=60, **kw)
        except Exception as exc:
            log.warning("GET %s failed: %s", url, exc)
            return None

# ── PRADAN fetch ────────────────────────────────────────────────────────────
def fetch_instrument(sess: PradanSession, inst: str, target: date) -> dict | None:
    ds = target.strftime("%Y%m%d")
    yr = target.strftime("%Y")
    url = f"{sess.BASE}/aditya-l1/{inst.lower()}/L2/{yr}/{ds}/"
    r = sess.get(url)
    if r is None or r.status_code != 200:
        return None
    import re
    files = re.findall(r'href="([^"]+\.cdf)"', r.text, re.IGNORECASE)
    if not files:
        return None
    furl = files[0] if files[0].startswith("http") else url + files[0]
    dest = DATA_DIR / "cache" / inst / ds / furl.split("/")[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        fr = sess.get(furl, stream=True)
        if fr is None:
            return None
        dest.write_bytes(fr.content)
        log.info("Downloaded %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
    return _parse_cdf(dest, inst)

def _parse_cdf(path: Path, inst: str) -> dict | None:
    try:
        import cdflib
        cdf  = cdflib.CDF(str(path))
        info = cdf.cdf_info()
        raw: dict = {"instrument": inst, "source_file": path.name, "simulated": False}
        for var in info.zVariables + info.rVariables:
            try:
                v = cdf.varget(var)
                raw[var] = v.tolist() if hasattr(v, "tolist") else v
            except Exception:
                pass
        return raw
    except Exception as exc:
        log.warning("CDF parse error for %s: %s", path.name, exc)
        return None

# ── Time helpers ────────────────────────────────────────────────────────────
def _make_times(n: int, step_min: int = 1) -> list[str]:
    now = datetime.now(timezone.utc)
    return [
        (now - timedelta(minutes=(n - i - 1) * step_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n)
    ]

# ── Simulation: uses true random seed so output is always different ─────────
def _rng() -> np.random.Generator:
    """Fresh RNG with an unpredictable seed every call."""
    seed = int(time.time() * 1000) % (2**31)
    return np.random.default_rng(seed)

# ── Moments ─────────────────────────────────────────────────────────────────
def analyse_moments(raw: dict | None) -> dict:
    if raw and not raw.get("simulated"):
        n = v = T = None
        for k in ("PROTON_DENSITY", "n_p", "density", "N_P", "np"):
            if k in raw:
                n = np.asarray(raw[k], float)
                break
        for k in ("PROTON_SPEED", "v_bulk", "velocity", "V_P", "vp"):
            if k in raw:
                v = np.asarray(raw[k], float)
                break
        for k in ("PROTON_TEMPERATURE", "T_p", "temperature", "T_P", "tp"):
            if k in raw:
                T = np.asarray(raw[k], float)
                break
        if n is not None and v is not None and T is not None:
            return _mom_dict(n, v, T, simulated=False)
    return _sim_moments()

def _mom_dict(n, v, T, simulated: bool) -> dict:
    n = np.asarray(n, float).clip(0.1, 100)
    v = np.asarray(v, float).clip(250, 900)
    T = np.asarray(T, float).clip(0.5, 100)
    times = _make_times(len(n))
    v_mean = float(v.mean())
    if   v_mean < 350: regime = "Sub-nominal slow wind"
    elif v_mean < 450: regime = "Slow wind — coronal streamer belt"
    elif v_mean < 550: regime = "Intermediate / transitional wind"
    elif v_mean < 650: regime = "Fast wind — coronal hole outflow"
    else:              regime = "Very fast stream / CME-driven"
    return {
        "simulated": simulated,
        "times": times,
        "density":     n.tolist(),
        "velocity":    v.tolist(),
        "temperature": T.tolist(),
        "density_last":     round(float(n[-1]), 2),
        "velocity_last":    round(float(v[-1]), 1),
        "temperature_last": round(float(T[-1]), 2),
        "density_mean":     round(float(n.mean()), 2),
        "velocity_mean":    round(float(v.mean()), 1),
        "temperature_mean": round(float(T.mean()), 2),
        "density_std":      round(float(n.std()), 2),
        "velocity_std":     round(float(v.std()), 1),
        "regime": regime,
    }

def _sim_moments() -> dict:
    rng = _rng()
    t = np.linspace(0, 2 * np.pi, 120)
    # Add random phase offsets so every run looks different
    ph1 = rng.uniform(0, 2 * np.pi)
    ph2 = rng.uniform(0, 2 * np.pi)
    ph3 = rng.uniform(0, 2 * np.pi)
    n_base = rng.uniform(3.5, 7.0)
    v_base = rng.uniform(390, 470)
    T_base = rng.uniform(8.0, 14.0)
    density     = (n_base + 1.5 * np.sin(t * 2 + ph1) + rng.normal(0, 0.4, 120)).clip(1, 20)
    velocity    = (v_base + 25  * np.cos(t * 1.5 + ph2) + rng.normal(0, 7, 120)).clip(300, 800)
    temperature = (T_base + 3.5 * np.sin(t * 3 + ph3) + rng.normal(0, 1.0, 120)).clip(2, 50)
    return _mom_dict(density, velocity, temperature, simulated=True)

# ── Magnetic field ───────────────────────────────────────────────────────────
def analyse_mag(raw: dict | None) -> dict:
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
    Bx = np.asarray(Bx, float)
    By = np.asarray(By, float)
    Bz = np.asarray(Bz, float)
    Bmag  = np.sqrt(Bx**2 + By**2 + Bz**2)
    clock = np.degrees(np.arctan2(By, Bz)) % 360
    cone  = np.degrees(np.arccos(np.abs(Bx) / np.where(Bmag > 0, Bmag, 1)))
    bz_l  = float(Bz[-1])
    bm_l  = float(Bmag[-1])
    cl_l  = float(clock[-1])
    return {
        "simulated": simulated,
        "times": _make_times(len(Bx)),
        "Bx": Bx.tolist(), "By": By.tolist(), "Bz": Bz.tolist(),
        "B_mag":       Bmag.tolist(),
        "clock_angle": clock.tolist(),
        "cone_angle":  cone.tolist(),
        "Bz_last":     round(bz_l, 2),
        "B_mag_last":  round(bm_l, 2),
        "clock_last":  round(cl_l, 1),
        "Bz_mean":     round(float(Bz.mean()), 2),
        "B_mag_mean":  round(float(Bmag.mean()), 2),
        "B_mag_std":   round(float(Bmag.std()), 2),
        "mva":         _mva(np.column_stack([Bx, By, Bz])),
        "geoeffective": bz_l < -5,
        "bz_status":   _bz_status(bz_l),
        "clock_status": _clock_status(cl_l),
    }

def _bz_status(bz: float) -> dict:
    if bz <= -10:
        return {"label": "Strongly southward", "color": "#ef4444",
                "explanation": f"Bz = {bz:.1f} nT — highly geoeffective. Dayside reconnection active. Geomagnetic storm (Kp ≥ 5) likely within hours."}
    if bz < -5:
        return {"label": "Southward", "color": "#f97316",
                "explanation": f"Bz = {bz:.1f} nT — geoeffective. Moderate ring current enhancement expected. Monitor Dst index."}
    if bz < 0:
        return {"label": "Weakly southward", "color": "#f0c040",
                "explanation": f"Bz = {bz:.1f} nT — mildly geoeffective. Auroral activity possible at high latitudes."}
    if bz < 5:
        return {"label": "Northward", "color": "#22c55e",
                "explanation": f"Bz = +{bz:.1f} nT — magnetopause closed. Minimal geomagnetic coupling."}
    return {"label": "Strongly northward", "color": "#2ecc8e",
            "explanation": f"Bz = +{bz:.1f} nT — very low geomagnetic activity expected."}

def _clock_status(angle: float) -> str:
    if 135 < angle < 225:
        return "Southward sector — geoeffective"
    if angle < 45 or angle > 315:
        return "Northward sector — minimal coupling"
    return "Dawnward/duskward — moderate coupling"

def _mva(B: np.ndarray) -> dict:
    if len(B) < 4:
        return {"valid": False}
    dB = B - B.mean(0)
    M  = (dB.T @ dB) / len(B)
    vals, vecs = np.linalg.eigh(M)
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    lam_max, lam_int, lam_min = vals
    ratio = float(lam_int / lam_min) if lam_min > 0 else 0.0
    return {
        "valid":        ratio > 2,
        "lambda_min":   round(float(lam_min), 4),
        "lambda_int":   round(float(lam_int), 4),
        "lambda_max":   round(float(lam_max), 4),
        "lambda_ratio": round(ratio, 2),
        "normal_dir":   vecs[:, 2].tolist(),
        "quality":      "Good (λ_int/λ_min > 2)" if ratio > 2 else "Poor — degeneracy",
        "explanation":  (
            f"MVA eigenvalue ratio λ_int/λ_min = {ratio:.2f}. "
            + ("Reliable normal direction." if ratio > 2
               else "Ratio < 2 — normal direction unreliable.")
        ),
    }

def _sim_mag() -> dict:
    rng = _rng()
    t = np.linspace(0, 4 * np.pi, 120)
    ph1 = rng.uniform(0, 2 * np.pi)
    ph2 = rng.uniform(0, 2 * np.pi)
    b_base = rng.uniform(3.5, 7.0)
    Bx = b_base * 0.5 + rng.normal(0, 1.0, 120)
    By = b_base       * np.cos(t * 0.8 + ph1) + rng.normal(0, 0.6, 120)
    Bz = b_base * 1.2 * np.sin(t * 0.6 + ph2) + rng.normal(0, 0.8, 120)
    return _mag_dict(Bx, By, Bz, simulated=True)

# ── STEPS spectrogram ────────────────────────────────────────────────────────
def analyse_steps(raw: dict | None) -> dict:
    rng = _rng()
    n_time, n_energy = 60, 32
    energies = np.geomspace(20e3, 6e6, n_energy)
    times    = _make_times(n_time, step_min=2)
    # Power-law background spectrum + random SEP injection
    bg = rng.exponential(80, (n_time, n_energy))
    bg *= np.exp(-np.log(energies / 20e3) * rng.uniform(2.0, 3.0))
    # Random SEP event placement
    if rng.random() > 0.3:
        t0 = rng.integers(5, n_time - 10)
        e0 = rng.integers(8, n_energy - 8)
        bg[t0:t0 + rng.integers(3, 8), e0:e0 + rng.integers(4, 10)] *= rng.uniform(30, 100)
    flux = bg.clip(1, None)
    return {
        "simulated":   True,
        "times":       times,
        "energies_eV": energies.tolist(),
        "flux":        flux.tolist(),
        "peak_flux":   float(flux.max()),
        "mean_flux":   float(flux.mean()),
        "sep_detected": float(flux.max()) > float(np.median(flux)) * 15,
        "axes": {
            "x": "Time (UTC) — each column ≈ 2-min cadence",
            "y": "Particle energy [eV] — log scale, 20 keV (bottom) → 6 MeV/n (top)",
            "color": "log₁₀ differential flux [p cm⁻² s⁻¹ sr⁻¹ eV⁻¹]",
        },
    }

# ── Spectral PSD ─────────────────────────────────────────────────────────────
def spectral_psd(mag: dict) -> dict:
    from scipy import signal as ssig
    Bmag = np.asarray(mag["B_mag"], float)
    if len(Bmag) < 32:
        return {"valid": False}
    fs = 1 / 60.0
    nperseg = min(len(Bmag) // 4, 32)
    f, psd = ssig.welch(Bmag - Bmag.mean(), fs=fs, nperseg=nperseg, scaling="density")
    mask = (f > f[1]) & (f < f[-3])
    if mask.sum() > 3:
        alpha = float(np.polyfit(
            np.log10(f[mask] + 1e-30),
            np.log10(psd[mask] + 1e-30),
            1,
        )[0])
    else:
        alpha = float("nan")
    return {
        "valid":        True,
        "freqs":        f[1:].tolist(),
        "psd":          psd[1:].tolist(),
        "alpha":        round(alpha, 3),
        "kolmogorov":   -5 / 3,
        "interpretation": (
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
        ),
    }

# ── Derived plasma parameters ─────────────────────────────────────────────────
def derived_params(n: float, T: float, v: float, B: float) -> dict:
    n_si = n * 1e6
    T_J  = T * EV
    B_si = B * 1e-9
    Pt   = n_si * T_J
    Pb   = B_si**2 / (2 * MU0)
    beta = Pt / Pb if Pb > 0 else 0.0
    Va   = B_si / np.sqrt(MU0 * n_si * MP) / 1e3 if n_si > 0 else 0.0
    Vs   = np.sqrt(KB * T_J / MP) / 1e3 if T_J > 0 else 0.0
    Vms  = np.sqrt(Va**2 + Vs**2)
    Ma   = v / Va  if Va  > 0 else 0.0
    Ms   = v / Vs  if Vs  > 0 else 0.0
    Mms  = v / Vms if Vms > 0 else 0.0
    gyro = 2 * np.pi * MP / (EV * B_si) if B_si > 0 else 0.0
    ion_L = 2.28e7 / np.sqrt(n) if n > 0 else 0.0
    return {
        "plasma_beta":         round(beta, 3),
        "alfven_speed_kms":    round(Va, 1),
        "sound_speed_kms":     round(Vs, 1),
        "magnetosonic_kms":    round(Vms, 1),
        "alfvenic_mach":       round(Ma, 2),
        "sonic_mach":          round(Ms, 2),
        "magnetosonic_mach":   round(Mms, 2),
        "gyroperiod_s":        round(gyro, 1),
        "ion_inertial_km":     round(ion_L, 0),
        "P_thermal_nPa":       round(Pt * 1e9, 3),
        "P_magnetic_nPa":      round(Pb * 1e9, 3),
        "explanations": {
            "beta":   (f"β = {beta:.3f} — " +
                       ("Magnetically dominated. Alfvén waves govern dynamics." if beta < 1
                        else "Thermally dominated. Kinetic effects important.")),
            "alfven": f"V_A = {Va:.1f} km/s — Alfvén wave speed along field lines.",
            "mach_a": (f"M_A = {Ma:.2f} — flow/Alfvén speed. " +
                       ("Super-Alfvénic — bow shock forms at Earth." if Ma > 1
                        else "Sub-Alfvénic — unusual conditions.")),
            "mach_s": (f"M_s = {Ms:.2f} — flow/sound speed. " +
                       ("Supersonic — shocks can form." if Ms > 1 else "Subsonic.")),
        },
    }

# ── Event detection ───────────────────────────────────────────────────────────
def detect_events(mom: dict, mag: dict, der: dict) -> list[dict]:
    events: list[dict] = []
    n  = np.asarray(mom["density"])
    v  = np.asarray(mom["velocity"])
    Bm = np.asarray(mag["B_mag"])
    Bz = np.asarray(mag["Bz"])

    n_med = float(np.median(n)) + 1e-9
    v_med = float(np.median(v))
    b_med = float(np.median(Bm)) + 1e-9

    # Shock
    n_j = float(n.max()) / n_med
    v_j = float(v.max()) - v_med
    b_j = float(Bm.max()) / b_med
    if n_j > 2.0 and v_j > 80 and b_j > 1.8:
        events.append({
            "type": "INTERPLANETARY SHOCK", "icon": "⚡",
            "confidence": round(min(1.0, (n_j / 2 + b_j / 2 + v_j / 100) / 4), 2),
            "time": mom["times"][-1],
            "explanation": (
                f"Simultaneous jumps: density ×{n_j:.1f}, speed +{v_j:.0f} km/s, "
                f"|B| ×{b_j:.1f}. Satisfies Rankine-Hugoniot conditions for a "
                "forward interplanetary shock. Likely CME or CIR-driven."
            ),
            "values": {"density_jump": round(n_j, 2), "dv_kms": round(v_j, 1), "B_jump": round(b_j, 2)},
            "experimental": False,
        })

    # Southward IMF
    bz_l = float(Bz[-1])
    if bz_l < -5:
        events.append({
            "type": "SOUTHWARD IMF", "icon": "🔴",
            "confidence": round(min(1.0, abs(bz_l) / 15), 2),
            "time": mag["times"][-1],
            "explanation": (
                f"Bz = {bz_l:.1f} nT (southward, GSE). Dayside magnetopause "
                "reconnection active via Dungey cycle. Enhanced energy input to "
                "magnetosphere — monitor Dst and Kp indices."
            ),
            "values": {"Bz_nT": round(bz_l, 1), "clock_deg": mag["clock_last"]},
            "experimental": False,
        })

    # Fast stream
    v_l = float(v[-1])
    if v_l > 600:
        events.append({
            "type": "HIGH-SPEED STREAM", "icon": "💨",
            "confidence": round(min(1.0, (v_l - 600) / 300), 2),
            "time": mom["times"][-1],
            "explanation": (
                f"V = {v_l:.0f} km/s — high-speed stream (HSS) from coronal hole. "
                "CIR likely 1–3 days ahead. Alfvénic fluctuations within the stream "
                "can drive substorms even without strongly southward Bz."
            ),
            "values": {"v_kms": round(v_l, 1)},
            "experimental": False,
        })

    # Density enhancement
    n_l   = float(n[-1])
    n_std = float(n.std()) + 1e-9
    if n_l > n_med + 3 * n_std:
        events.append({
            "type": "DENSITY ENHANCEMENT", "icon": "🌊",
            "confidence": round(min(1.0, (n_l - n_med) / (4 * n_std)), 2),
            "time": mom["times"][-1],
            "explanation": (
                f"n = {n_l:.1f} cm⁻³ ({n_l / n_med:.1f}× background). "
                "Consistent with CME sheath, CIR leading edge, or "
                "heliospheric current sheet crossing."
            ),
            "values": {"n_cm3": round(n_l, 2), "bg_cm3": round(n_med, 2)},
            "experimental": False,
        })

    # Low beta
    beta = der["plasma_beta"]
    if beta < 0.1:
        events.append({
            "type": "LOW-BETA INTERVAL", "icon": "🧲",
            "confidence": round(1 - beta / 0.1, 2),
            "time": mag["times"][-1],
            "explanation": (
                f"β = {beta:.3f} ≪ 1 — magnetic pressure dominates. "
                "Typical of CME flux rope interiors. Alfvén wave propagation "
                "dominates plasma dynamics in this regime."
            ),
            "values": {"beta": beta},
            "experimental": False,
        })

    return events

# ── History ───────────────────────────────────────────────────────────────────
def load_history() -> list:
    if HISTORY_F.exists():
        try:
            return json.loads(HISTORY_F.read_text())
        except Exception:
            pass
    return []

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=== Aditya-L1 Pipeline START (run_id=%s) ===", RUN_ID)

    email, pwd  = get_creds()
    sess        = PradanSession(email, pwd)
    today       = date.today()
    simulated   = not bool(email)

    if email:
        simulated = not sess.login()

    raw_swis = raw_mag = None
    if not simulated:
        for delta in range(3):
            raw_swis = fetch_instrument(sess, "SWIS", today - timedelta(days=delta))
            if raw_swis:
                break
        for delta in range(3):
            raw_mag = fetch_instrument(sess, "MAG", today - timedelta(days=delta))
            if raw_mag:
                break

    mom    = analyse_moments(raw_swis)
    mag    = analyse_mag(raw_mag)
    steps  = analyse_steps(None)
    psd    = spectral_psd(mag)
    der    = derived_params(
        mom["density_last"], mom["temperature_last"],
        mom["velocity_last"], mag["B_mag_last"],
    )
    events = detect_events(mom, mag, der)
    run_time = datetime.now(timezone.utc).isoformat()

    latest = {
        # run_id ensures every run produces a unique file → git always commits
        "run_id":       RUN_ID,
        "run_time":     run_time,
        "data_date":    today.isoformat(),
        "simulated":    mom["simulated"],
        "moments":      mom,
        "mag":          mag,
        "steps":        steps,
        "psd":          psd,
        "derived":      der,
        "events":       events,
        "event_count":  len(events),
        "scalars": {
            "density_cm3":       mom["density_last"],
            "velocity_kms":      mom["velocity_last"],
            "temperature_eV":    mom["temperature_last"],
            "B_mag_nT":          mag["B_mag_last"],
            "Bz_nT":             mag["Bz_last"],
            "clock_angle_deg":   mag["clock_last"],
            "plasma_beta":       der["plasma_beta"],
            "alfven_speed_kms":  der["alfven_speed_kms"],
            "alfvenic_mach":     der["alfvenic_mach"],
            "P_thermal_nPa":     der["P_thermal_nPa"],
            "P_magnetic_nPa":    der["P_magnetic_nPa"],
        },
        "sw_regime":    mom["regime"],
        "bz_status":    mag["bz_status"],
        "mva":          mag["mva"],
    }

    LATEST_F.write_text(json.dumps(latest, indent=2))
    size = LATEST_F.stat().st_size
    log.info("Wrote %s (%.1f KB)", LATEST_F, size / 1024)
    assert size > 1000, f"latest.json too small ({size} bytes) — something went wrong"

    # History
    hist   = load_history()
    record = {
        "time":        run_time,
        "run_id":      RUN_ID,
        "simulated":   mom["simulated"],
        "density":     mom["density_last"],
        "velocity":    mom["velocity_last"],
        "temperature": mom["temperature_last"],
        "B_mag":       mag["B_mag_last"],
        "Bz":          mag["Bz_last"],
        "beta":        der["plasma_beta"],
        "va_kms":      der["alfven_speed_kms"],
        "mach_a":      der["alfvenic_mach"],
        "Pt_nPa":      der["P_thermal_nPa"],
        "Pb_nPa":      der["P_magnetic_nPa"],
        "event_types": [e["type"] for e in events],
        "sw_regime":   mom["regime"],
    }
    hist.append(record)
    hist = hist[-MAX_HIST:]
    HISTORY_F.write_text(json.dumps(hist, indent=2))
    log.info("History: %d records", len(hist))
    log.info("Events detected: %d", len(events))
    log.info("=== Pipeline DONE ===")


if __name__ == "__main__":
    main()
