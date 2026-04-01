"""
scripts/fetch_and_process.py
============================
Aditya-L1 in-situ plasma data pipeline.
Run by GitHub Actions every 6 hours.
Credentials come from GitHub Secrets — never hardcoded.

Outputs
-------
data/latest.json   — full analysis + explanations for dashboard
data/history.json  — rolling 6-month record
"""
from __future__ import annotations
import json, logging, os, time, warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = Path("data")
LATEST_F  = DATA_DIR / "latest.json"
HISTORY_F = DATA_DIR / "history.json"
MAX_HIST  = 720
DATA_DIR.mkdir(exist_ok=True)

MP  = 1.67262192e-27
KB  = 1.38064852e-23
EV  = 1.60217663e-19
MU0 = 4*np.pi*1e-7

def get_creds():
    e = os.getenv("PRADAN_EMAIL","")
    p = os.getenv("PRADAN_PASSWORD","")
    if not e or not p:
        log.warning("No PRADAN credentials → simulation mode")
    return e, p

class PradanSession:
    BASE="https://pradan.issdc.gov.in"
    def __init__(self,e,p): self.e=e; self.p=p; self.s=None; self._t=0.0
    def _exp(self): return self.s is None or (time.time()-self._t)>25*60
    def login(self):
        if not self.e: return False
        try:
            self.s=requests.Session()
            r=self.s.post(f"{self.BASE}/login",data={"email":self.e,"password":self.p},timeout=30)
            r.raise_for_status(); self._t=time.time()
            log.info("PRADAN login OK"); return True
        except Exception as ex: log.warning("Login fail: %s",ex); self.s=None; return False
    def get(self,url,**kw):
        if self._exp(): self.login()
        if not self.s: return None
        try: return self.s.get(url,timeout=60,**kw)
        except Exception as ex: log.warning("GET fail: %s",ex); return None

def fetch_inst(sess,inst,target):
    ds=target.strftime("%Y%m%d"); yr=target.strftime("%Y")
    url=f"{sess.BASE}/aditya-l1/{inst.lower()}/L2/{yr}/{ds}/"
    r=sess.get(url)
    if r is None or r.status_code!=200: return None
    import re
    files=re.findall(r'href="([^"]+\.cdf)"',r.text,re.I)
    if not files: return None
    furl=files[0] if files[0].startswith("http") else url+files[0]
    dest=DATA_DIR/"cache"/inst/ds/furl.split("/")[-1]
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        fr=sess.get(furl,stream=True)
        if fr is None: return None
        dest.write_bytes(fr.content)
    return parse_cdf(dest,inst)

def parse_cdf(path,inst):
    try:
        import cdflib
        cdf=cdflib.CDF(str(path)); info=cdf.cdf_info()
        raw={"instrument":inst,"source_file":path.name,"simulated":False}
        for var in info.zVariables+info.rVariables:
            try:
                v=cdf.varget(var); raw[var]=v.tolist() if hasattr(v,"tolist") else v
            except: pass
        return raw
    except Exception as ex: log.warning("CDF err: %s",ex); return None

def make_times(n,step_min=1):
    now=datetime.now(timezone.utc)
    return [(now-timedelta(minutes=(n-i-1)*step_min)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]

# ── Moments ───────────────────────────────────────────────────────────────────
def analyse_moments(raw):
    if raw and not raw.get("simulated"):
        n=v=T=None
        for k in ("PROTON_DENSITY","n_p","density","N_P","np"):
            if k in raw: n=np.asarray(raw[k],float); break
        for k in ("PROTON_SPEED","v_bulk","velocity","V_P","vp"):
            if k in raw: v=np.asarray(raw[k],float); break
        for k in ("PROTON_TEMPERATURE","T_p","temperature","T_P","tp"):
            if k in raw: T=np.asarray(raw[k],float); break
        if n is not None and v is not None and T is not None:
            return _mom(n,v,T,False)
    return _sim_mom()

def _mom(n,v,T,sim):
    n=np.asarray(n,float).clip(0.1,100)
    v=np.asarray(v,float).clip(250,900)
    T=np.asarray(T,float).clip(0.5,100)
    t=make_times(len(n))
    regime=("Sub-nominal slow wind" if v.mean()<350 else
            "Slow wind — coronal streamer belt" if v.mean()<450 else
            "Intermediate / transitional wind" if v.mean()<550 else
            "Fast wind — coronal hole outflow" if v.mean()<650 else
            "Very fast stream / CME-driven")
    return {"simulated":sim,"times":t,
            "density":n.tolist(),"velocity":v.tolist(),"temperature":T.tolist(),
            "density_last":round(float(n[-1]),2),"velocity_last":round(float(v[-1]),1),
            "temperature_last":round(float(T[-1]),2),"density_mean":round(float(n.mean()),2),
            "velocity_mean":round(float(v.mean()),1),"temperature_mean":round(float(T.mean()),2),
            "density_std":round(float(n.std()),2),"velocity_std":round(float(v.std()),1),
            "regime":regime}

def _sim_mom():
    rng=np.random.default_rng(int(time.time())//600)
    t=np.linspace(0,2*np.pi,120)
    n=(5+1.2*np.sin(t*2+.3)+rng.normal(0,.5,120)).clip(1,20)
    v=(430+20*np.cos(t*1.5+1)+rng.normal(0,8,120)).clip(300,800)
    T=(10+3*np.sin(t*3)+rng.normal(0,1.2,120)).clip(2,50)
    return _mom(n,v,T,True)

# ── Magnetic field ────────────────────────────────────────────────────────────
def analyse_mag(raw):
    if raw and not raw.get("simulated"):
        Bx=By=Bz=None
        for k in ("BX_GSE","Bx","B_X","BX","bx"):
            if k in raw: Bx=np.asarray(raw[k],float); break
        for k in ("BY_GSE","By","B_Y","BY","by"):
            if k in raw: By=np.asarray(raw[k],float); break
        for k in ("BZ_GSE","Bz","B_Z","BZ","bz"):
            if k in raw: Bz=np.asarray(raw[k],float); break
        if Bx is not None and By is not None and Bz is not None:
            return _mag(Bx,By,Bz,False)
    return _sim_mag()

def _mag(Bx,By,Bz,sim):
    Bx=np.asarray(Bx,float); By=np.asarray(By,float); Bz=np.asarray(Bz,float)
    Bm=np.sqrt(Bx**2+By**2+Bz**2)
    clock=np.degrees(np.arctan2(By,Bz))%360
    cone=np.degrees(np.arccos(np.abs(Bx)/np.where(Bm>0,Bm,1)))
    bz_l=float(Bz[-1]); bm_l=float(Bm[-1]); cl_l=float(clock[-1])
    bz_st=(_bz_status(bz_l))
    mva=_mva(np.column_stack([Bx,By,Bz]))
    return {"simulated":sim,"times":make_times(len(Bx)),
            "Bx":Bx.tolist(),"By":By.tolist(),"Bz":Bz.tolist(),
            "B_mag":Bm.tolist(),"clock_angle":clock.tolist(),"cone_angle":cone.tolist(),
            "Bz_last":round(bz_l,2),"B_mag_last":round(bm_l,2),
            "clock_last":round(cl_l,1),"cone_last":round(float(cone[-1]),1),
            "Bz_mean":round(float(Bz.mean()),2),"B_mag_mean":round(float(Bm.mean()),2),
            "B_mag_std":round(float(Bm.std()),2),
            "mva":mva,"geoeffective":bz_l<-5,"bz_status":bz_st,
            "clock_status":_clock_status(cl_l)}

def _bz_status(bz):
    if bz<=-10: return{"label":"Strongly southward","color":"#ef4444",
        "explanation":f"Bz={bz:.1f} nT — highly geoeffective. Strong dayside reconnection; geomagnetic storm (Kp≥5) likely within hours."}
    if bz<-5:   return{"label":"Southward","color":"#f97316",
        "explanation":f"Bz={bz:.1f} nT — geoeffective. Moderate ring current enhancement expected. Monitor Dst index."}
    if bz<0:    return{"label":"Weakly southward","color":"#fbbf24",
        "explanation":f"Bz={bz:.1f} nT — mildly geoeffective. Auroral activity possible at high latitudes."}
    if bz<5:    return{"label":"Northward","color":"#22c55e",
        "explanation":f"Bz=+{bz:.1f} nT — magnetopause closed. Minimal geomagnetic coupling."}
    return       {"label":"Strongly northward","color":"#34d399",
        "explanation":f"Bz=+{bz:.1f} nT — very low geomagnetic activity expected."}

def _clock_status(a):
    if 135<a<225: return "Southward sector — geoeffective"
    if a<45 or a>315: return "Northward sector — minimal coupling"
    return "Dawnward/duskward — moderate coupling"

def _mva(B):
    if len(B)<4: return{"valid":False}
    dB=B-B.mean(0); M=(dB.T@dB)/len(B)
    vals,vecs=np.linalg.eigh(M)
    idx=np.argsort(vals)[::-1]; vals=vals[idx]; vecs=vecs[:,idx]
    lam_max,lam_int,lam_min=vals
    r=float(lam_int/lam_min) if lam_min>0 else 0
    return{"valid":r>2,"lambda_min":round(float(lam_min),4),
           "lambda_int":round(float(lam_int),4),"lambda_max":round(float(lam_max),4),
           "lambda_ratio":round(r,2),"normal_dir":vecs[:,2].tolist(),
           "quality":"Good (λ_int/λ_min>2)" if r>2 else "Poor — degeneracy",
           "explanation":f"MVA ratio λ_int/λ_min={r:.2f}. "+
               ("Reliable normal direction." if r>2 else "Ratio<2 — normal unreliable.")}

def _sim_mag():
    rng=np.random.default_rng(int(time.time())//600+1); n=120
    t=np.linspace(0,4*np.pi,n)
    Bx=2+rng.normal(0,1.2,n); By=3*np.cos(t*.8)+rng.normal(0,.8,n)
    Bz=4*np.sin(t*.6+.5)+rng.normal(0,1.,n)
    return _mag(Bx,By,Bz,True)

# ── STEPS spectrogram ─────────────────────────────────────────────────────────
def analyse_steps(raw):
    rng=np.random.default_rng(int(time.time())//600+2)
    nT,nE=60,32
    E=np.geomspace(20e3,6e6,nE)
    times=make_times(nT,step_min=2)
    bg=rng.exponential(100,(nT,nE))*np.exp(-np.log(E/20e3)*2.5)
    bg[18:25,12:22]*=60; bg[10:14,8:14]*=15
    flux=bg.clip(1,None)
    return{"simulated":True,"times":times,"energies_eV":E.tolist(),
           "flux":flux.tolist(),"peak_flux":float(flux.max()),
           "mean_flux":float(flux.mean()),
           "sep_detected":float(flux.max())>float(np.median(flux))*20,
           "axes":{"x":"Time (UTC) — measurement cadence ~2 min",
                   "y":"Particle energy [eV] — log scale, 20 keV to 6 MeV/n",
                   "color":"log₁₀ differential flux [particles cm⁻² s⁻¹ sr⁻¹ eV⁻¹]",
                   "bright":"Bright streaks = solar energetic particle (SEP) events"}}

# ── Spectral PSD ──────────────────────────────────────────────────────────────
def spectral_psd(mag):
    from scipy import signal
    Bm=np.asarray(mag["B_mag"],float)
    if len(Bm)<32: return{"valid":False}
    fs=1/60; nperseg=min(len(Bm)//4,32)
    f,psd=signal.welch(Bm-Bm.mean(),fs=fs,nperseg=nperseg,scaling="density")
    mask=(f>f[1])&(f<f[-3])
    alpha=float(np.polyfit(np.log10(f[mask]+1e-30),np.log10(psd[mask]+1e-30),1)[0]) if mask.sum()>3 else float("nan")
    return{"valid":True,"freqs":f[1:].tolist(),"psd":psd[1:].tolist(),
           "alpha":round(alpha,3),"kolmogorov":-5/3,
           "axes":{"x":"Frequency f [Hz] — spacecraft-frame fluctuation frequency",
                   "y":"PSD [(nT)² Hz⁻¹] — power spectral density of |B| fluctuations",
                   "kolmogorov_line":"Kolmogorov −5/3 reference (MHD inertial-range turbulence)",
                   "IK_line":"Iroshnikov-Kraichnan −3/2 reference"},
           "interpretation":f"Spectral index α={alpha:.2f}. Kolmogorov MHD turbulence predicts −1.67. "
               +("Steeper → sub-ion dissipation / kinetic Alfvén waves." if alpha<-2
                 else "Shallower → energy injection from shocks or CMEs." if not np.isnan(alpha) and alpha>-1.2
                 else "Near Kolmogorov — typical inertial-range turbulence." if not np.isnan(alpha) else "")}

# ── Derived plasma parameters ─────────────────────────────────────────────────
def derived_params(n,T,v,B):
    n_si=n*1e6; T_J=T*EV; B_si=B*1e-9
    Pt=n_si*T_J; Pb=B_si**2/(2*MU0)
    beta=Pt/Pb if Pb>0 else 0.0
    Va=B_si/np.sqrt(MU0*n_si*MP)/1e3 if n_si>0 else 0
    Vs=np.sqrt(KB*T_J/MP)/1e3 if T_J>0 else 0
    Vms=np.sqrt(Va**2+Vs**2)
    Ma=v/Va if Va>0 else 0
    Ms=v/Vs if Vs>0 else 0
    Mms=v/Vms if Vms>0 else 0
    gyro=2*np.pi*MP/(EV*B_si) if B_si>0 else 0
    ion_L=2.28e7/np.sqrt(n) if n>0 else 0
    return{"plasma_beta":round(beta,3),"alfven_speed_kms":round(Va,1),
           "sound_speed_kms":round(Vs,1),"magnetosonic_kms":round(Vms,1),
           "alfvenic_mach":round(Ma,2),"sonic_mach":round(Ms,2),
           "magnetosonic_mach":round(Mms,2),"gyroperiod_s":round(gyro,1),
           "ion_inertial_km":round(ion_L,0),"P_thermal_nPa":round(Pt*1e9,3),
           "P_magnetic_nPa":round(Pb*1e9,3),
           "explanations":{
               "beta":f"β={beta:.3f} — thermal/magnetic pressure ratio. "+
                   ("Magnetically dominated (β<1) — Alfvén modes propagate." if beta<1 else
                    "Thermally dominated (β>1) — kinetic effects important."),
               "alfven":f"V_A={Va:.1f} km/s — Alfvén wave speed along field lines.",
               "mach_a":f"M_A={Ma:.2f} — flow/Alfvén speed ratio. "+
                   ("Super-Alfvénic — bow shock forms." if Ma>1 else "Sub-Alfvénic — unusual."),
               "mach_s":f"M_s={Ms:.2f} — flow/sound speed ratio. "+
                   ("Supersonic — shocks possible." if Ms>1 else "Subsonic.")}}

# ── Event detection ───────────────────────────────────────────────────────────
def detect_events(mom,mag,der):
    events=[]
    n=np.asarray(mom["density"]); v=np.asarray(mom["velocity"])
    Bm=np.asarray(mag["B_mag"]); Bz=np.asarray(mag["Bz"])
    n_med=float(np.median(n)); v_med=float(np.median(v))
    b_med=float(np.median(Bm))
    n_j=float(n.max())/(n_med+1e-9); v_j=float(v.max())-v_med; b_j=float(Bm.max())/(b_med+1e-9)
    if n_j>2.0 and v_j>80 and b_j>1.8:
        conf=min(1.0,(n_j/2+b_j/2+v_j/100)/4)
        events.append({"type":"INTERPLANETARY SHOCK","icon":"⚡","confidence":round(conf,2),
            "time":mom["times"][-1],
            "explanation":f"Density ×{n_j:.1f}, speed +{v_j:.0f} km/s, |B| ×{b_j:.1f} — satisfies Rankine-Hugoniot conditions for a forward interplanetary shock. Likely CME or CIR-driven.",
            "values":{"density_jump":round(n_j,2),"dv_kms":round(v_j,1),"B_jump":round(b_j,2)},"experimental":False})
    bz_l=float(Bz[-1])
    if bz_l<-5:
        events.append({"type":"SOUTHWARD IMF","icon":"🔴","confidence":round(min(1.0,abs(bz_l)/15),2),
            "time":mag["times"][-1],
            "explanation":f"Bz={bz_l:.1f} nT — southward in GSE frame. Dayside magnetopause reconnection active. Reconnection rate ∝ V_sw·B_T·sin²(θ/2). Enhanced energy input to magnetosphere; monitor Dst and Kp.",
            "values":{"Bz_nT":bz_l,"clock_deg":mag["clock_last"]},"experimental":False})
    v_l=float(v[-1])
    if v_l>600:
        events.append({"type":"HIGH-SPEED STREAM","icon":"💨","confidence":round(min(1.0,(v_l-600)/300),2),
            "time":mom["times"][-1],
            "explanation":f"V_sw={v_l:.0f} km/s — high-speed stream (HSS) from coronal hole. CIR arrives 1–3 days later. Alfvénic fluctuations within stream can drive substorms even without strong southward Bz.",
            "values":{"v_kms":round(v_l,1)},"experimental":False})
    n_l=float(n[-1]); n_std=float(n.std())+1e-9
    if n_l>n_med+3*n_std:
        events.append({"type":"DENSITY ENHANCEMENT","icon":"🌊","confidence":round(min(1.0,(n_l-n_med)/(4*n_std)),2),
            "time":mom["times"][-1],
            "explanation":f"n={n_l:.1f} cm⁻³ ({n_l/n_med:.1f}× background). Consistent with CME sheath, CIR leading edge, or sector boundary crossing. Cold + dense + slow → classic ICME ejecta.",
            "values":{"n_cm3":round(n_l,2),"bg_cm3":round(n_med,2)},"experimental":False})
    beta=der["plasma_beta"]
    if beta<0.1:
        events.append({"type":"LOW-BETA INTERVAL","icon":"🧲","confidence":round(1-beta/0.1,2),
            "time":mag["times"][-1],
            "explanation":f"β={beta:.3f} ≪ 1 — magnetic pressure dominates thermal pressure. Typical of flux rope interiors (CME ejecta) or near-Sun fast wind. Alfvén wave propagation dominates plasma dynamics.",
            "values":{"beta":beta},"experimental":False})
    return events

def load_hist():
    if HISTORY_F.exists():
        try: return json.loads(HISTORY_F.read_text())
        except: pass
    return []

def main():
    log.info("=== Aditya-L1 Pipeline START ===")
    email,pwd=get_creds()
    sess=PradanSession(email,pwd)
    today=date.today(); sim=not bool(email)
    if email: sim=not sess.login()
    raw_swis=raw_mag=None
    if not sim:
        for d in range(3):
            raw_swis=fetch_inst(sess,"SWIS",today-timedelta(days=d))
            if raw_swis: break
        for d in range(3):
            raw_mag=fetch_inst(sess,"MAG",today-timedelta(days=d))
            if raw_mag: break
    mom=analyse_moments(raw_swis)
    mag=analyse_mag(raw_mag)
    steps=analyse_steps(None)
    psd=spectral_psd(mag)
    der=derived_params(mom["density_last"],mom["temperature_last"],mom["velocity_last"],mag["B_mag_last"])
    events=detect_events(mom,mag,der)
    run_time=datetime.now(timezone.utc).isoformat()
    latest={"run_time":run_time,"data_date":today.isoformat(),"simulated":mom["simulated"],
            "moments":mom,"mag":mag,"steps":steps,"psd":psd,"derived":der,
            "events":events,"event_count":len(events),
            "scalars":{"density_cm3":mom["density_last"],"velocity_kms":mom["velocity_last"],
                       "temperature_eV":mom["temperature_last"],"B_mag_nT":mag["B_mag_last"],
                       "Bz_nT":mag["Bz_last"],"clock_angle_deg":mag["clock_last"],
                       "plasma_beta":der["plasma_beta"],"alfven_speed_kms":der["alfven_speed_kms"],
                       "alfvenic_mach":der["alfvenic_mach"],"P_thermal_nPa":der["P_thermal_nPa"],
                       "P_magnetic_nPa":der["P_magnetic_nPa"]},
            "sw_regime":mom["regime"],"bz_status":mag["bz_status"],"mva":mag["mva"]}
    LATEST_F.write_text(json.dumps(latest,indent=2))
    hist=load_hist()
    rec={"time":run_time,"simulated":mom["simulated"],"density":mom["density_last"],
         "velocity":mom["velocity_last"],"temperature":mom["temperature_last"],
         "B_mag":mag["B_mag_last"],"Bz":mag["Bz_last"],"beta":der["plasma_beta"],
         "va_kms":der["alfven_speed_kms"],"mach_a":der["alfvenic_mach"],
         "Pt_nPa":der["P_thermal_nPa"],"Pb_nPa":der["P_magnetic_nPa"],
         "event_types":[e["type"] for e in events],"sw_regime":mom["regime"]}
    hist.append(rec); hist=hist[-MAX_HIST:]
    HISTORY_F.write_text(json.dumps(hist,indent=2))
    log.info("Done. %d history records, %d events.",len(hist),len(events))

if __name__=="__main__": main()
