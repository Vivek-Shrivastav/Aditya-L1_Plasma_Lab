/**
 * Aditya-L1 Plasma Lab — Main Application
 *
 * Data source priority:
 *  1. data/latest.json  — Aditya-L1 pipeline (rich scalars, updated every 6 h via GitHub Actions)
 *  2. NOAA SWPC DSCOVR  — 7-day time-series for real-time charting (updated ~1 min)
 *  3. Simulated data    — physics-realistic fallback when both sources are unreachable
 */

// ── State ─────────────────────────────────────────────────────────────────────
const STATE = {
    currentView: 'home',
    noaa: { plasma: [], mag: [] },   // NOAA SWPC 7-day time-series
    pipeline: null,                   // data/latest.json (Aditya-L1 pipeline)
    history: [],                      // data/history.json (pipeline history)
    status: 'offline',
    lastUpdate: null,
    config: {
        refreshInterval: 60_000,      // ms between polls
        retryBase: 5_000,
        retryMax: 120_000,
        missionT0: new Date('2023-09-02T11:50:00+05:30').getTime()
    },
    _retryDelay: 5_000
};

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt0 = v => (v == null || isNaN(v) ? '—' : (+v).toFixed(0));
const fmt1 = v => (v == null || isNaN(v) ? '—' : (+v).toFixed(1));
const fmt2 = v => (v == null || isNaN(v) ? '—' : (+v).toFixed(2));

function setEl(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
}

// ── Data Fetching ─────────────────────────────────────────────────────────────
async function fetchAll() {
    setStatus('fetching');

    let noaaOk = false;
    let pipelineOk = false;

    // 1. NOAA SWPC — high-cadence 7-day time-series (always try)
    try {
        const bust = '?t=' + Date.now();
        const [pRes, mRes] = await Promise.all([
            fetch('https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json' + bust),
            fetch('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json' + bust)
        ]);
        if (pRes.ok && mRes.ok) {
            const plasma = await pRes.json();
            const mag    = await mRes.json();

            STATE.noaa.plasma = plasma.slice(1)
                .map(r => ({
                    time:    new Date(r[0]),
                    density: r[1] != null ? +r[1] : null,
                    speed:   r[2] != null ? +r[2] : null,
                    temp:    r[3] != null ? +r[3] : null
                }))
                .filter(d => d.density != null);

            STATE.noaa.mag = mag.slice(1)
                .map(r => ({
                    time: new Date(r[0]),
                    bx: +r[1], by: +r[2], bz: +r[3], bt: +r[6]
                }))
                .filter(d => !isNaN(d.bz));

            noaaOk = true;
        }
    } catch (e) {
        console.warn('[NOAA] fetch failed:', e.message);
    }

    // 2. Aditya-L1 pipeline — rich scalars + history
    try {
        const bust = '?t=' + Date.now();
        const [lRes, hRes] = await Promise.all([
            fetch('data/latest.json' + bust),
            fetch('data/history.json' + bust)
        ]);
        if (lRes.ok) {
            STATE.pipeline = await lRes.json();
            pipelineOk = true;
        }
        if (hRes.ok) {
            STATE.history = await hRes.json();
        }
    } catch (e) {
        console.warn('[Pipeline] fetch failed:', e.message);
    }

    // 3. Neither source worked — generate mock data
    if (!noaaOk && !pipelineOk) {
        generateMockData();
        setStatus('simulated');
    } else {
        STATE.lastUpdate = new Date();
        STATE._retryDelay = STATE.config.retryBase; // reset backoff on success
        const isSimulated = !noaaOk || STATE.pipeline?.simulated === true;
        setStatus(isSimulated ? 'simulated' : 'live');
    }

    onDataReceived();

    // Schedule next refresh
    setTimeout(fetchAll, STATE.config.refreshInterval);
}

// Generates physically realistic mock solar wind data as an offline fallback.
function generateMockData() {
    const now = Date.now();
    STATE.noaa.plasma = [];
    STATE.noaa.mag    = [];
    for (let i = 0; i < 200; i++) {
        const t = new Date(now - (200 - i) * 60_000);
        STATE.noaa.plasma.push({
            time:    t,
            density: 5 + Math.random() * 3,
            speed:   400 + Math.random() * 100,
            temp:    50000 + Math.random() * 20000
        });
        STATE.noaa.mag.push({
            time: t,
            bx: Math.random() * 4 - 2,
            by: Math.random() * 4 - 2,
            bz: Math.random() * 6 - 3,
            bt: 5 + Math.random() * 2
        });
    }
    STATE.lastUpdate = new Date();
}

// ── Status Display ────────────────────────────────────────────────────────────
function setStatus(status) {
    STATE.status = status;
    const dot  = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    if (!dot || !text) return;
    dot.className  = 'status-dot status-' + status;
    const LABELS = {
        live:      '● LIVE',
        simulated: '◑ SIMULATED',
        fetching:  '○ UPDATING…',
        offline:   '○ OFFLINE',
        error:     '✕ ERROR'
    };
    text.textContent = LABELS[status] || status.toUpperCase();
}

// ── Navigation ────────────────────────────────────────────────────────────────
function switchView(viewId) {
    STATE.currentView = viewId;
    document.querySelectorAll('.nav-link').forEach(l =>
        l.classList.toggle('active', l.dataset.view === viewId)
    );
    document.querySelectorAll('.view').forEach(v =>
        v.classList.toggle('active', v.id === viewId + '-view')
    );
    // Close mobile menu if open
    const mobileMenu = document.querySelector('.nav-links');
    mobileMenu?.classList.remove('open');

    setTimeout(() => {
        if (viewId === 'dashboard')  Dashboard.render();
        if (viewId === 'visualizer') Visualizer.render();
        if (viewId === 'lab')        Lab.render();
        if (viewId === 'history')    HistoryView.render();
    }, 50);
}

// ── Data Received Callback ────────────────────────────────────────────────────
function onDataReceived() {
    updateSnapshot();
    updateTimestamps();
    if (STATE.currentView !== 'home') switchView(STATE.currentView);
}

function updateTimestamps() {
    if (!STATE.lastUpdate) return;
    const timeStr = STATE.lastUpdate.toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    document.querySelectorAll('.last-update').forEach(el => (el.textContent = timeStr));

    if (STATE.pipeline?.run_time) {
        const ageMin = Math.round((Date.now() - new Date(STATE.pipeline.run_time)) / 60_000);
        const ageStr = ageMin < 60
            ? `${ageMin} min ago`
            : `${Math.round(ageMin / 60)} h ago`;
        setEl('data-age', ageStr);
    }
}

function updateSnapshot() {
    // Prefer pipeline scalars; fall back to NOAA latest
    const sc        = STATE.pipeline?.scalars;
    const pLast     = STATE.noaa.plasma[STATE.noaa.plasma.length - 1];
    const mLast     = STATE.noaa.mag[STATE.noaa.mag.length - 1];

    const density   = sc?.density_cm3      ?? pLast?.density;
    const speed     = sc?.velocity_kms     ?? pLast?.speed;
    const tempEV    = sc?.temperature_eV   ?? (pLast ? pLast.temp / 11605 : null); // K → eV
    const bmag      = sc?.B_mag_nT         ?? mLast?.bt;
    const bz        = sc?.Bz_nT            ?? mLast?.bz;

    setEl('snap-density', fmt1(density));
    setEl('snap-speed',   fmt0(speed));
    setEl('snap-temp',    fmt1(tempEV));
    setEl('snap-mag',     fmt1(bmag));
    setEl('snap-bz',      fmt1(bz));

    // Extended pipeline-only scalars
    if (sc) {
        setEl('snap-beta',  fmt2(sc.plasma_beta));
        setEl('snap-va',    fmt0(sc.alfven_speed_kms));
        setEl('snap-clock', fmt0(sc.clock_angle_deg));
        setEl('snap-mach',  fmt1(sc.alfvenic_mach));
        setEl('snap-pt',    fmt2(sc.P_thermal_nPa));
    }

    // Solar-wind regime & Bz status (pipeline)
    if (STATE.pipeline) {
        setEl('snap-regime', STATE.pipeline.sw_regime || '—');

        const bzStatus = STATE.pipeline.bz_status;
        if (bzStatus) {
            const el = $('badge-bz-status');
            if (el) {
                el.textContent = bzStatus.label;
                el.title       = bzStatus.explanation || '';
                el.className   = 'snapshot-badge badge-' + (bzStatus.color === 'red' ? 'alert' : bzStatus.color === 'yellow' ? 'warn' : 'ok');
            }
        }

        // Show/hide "SIMULATED DATA" banner
        const simBanner = $('sim-banner');
        if (simBanner) simBanner.style.display = STATE.pipeline.simulated ? 'block' : 'none';
    }

    // Threshold badges
    setBadge('badge-density', density > 15 ? 'Very High' : density > 10 ? 'Elevated'        : 'Normal',
                               density > 10 ? (density > 15 ? 'alert' : 'warn') : 'ok');
    setBadge('badge-speed',   speed > 600  ? 'Extreme'   : speed > 500 ? 'High-Speed Stream' : 'Normal',
                               speed > 500  ? (speed > 600 ? 'alert' : 'warn') : 'ok');
    setBadge('badge-bz',      bz < -10     ? 'Storm Warning' : bz < -5 ? 'Southward (active)' : bz > 5 ? 'Northward' : 'Stable',
                               bz < -5     ? (bz < -10 ? 'alert' : 'warn') : 'ok');

    // Raw JSON panel (Advanced section)
    const rawEl = $('raw-json-content');
    if (rawEl && STATE.pipeline) {
        rawEl.textContent = JSON.stringify(STATE.pipeline, null, 2);
    }
}

function setBadge(id, text, level) {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className   = 'snapshot-badge badge-' + (level || 'ok');
}

// ── Mission Elapsed-Time Timer ────────────────────────────────────────────────
function startMissionTimer() {
    const timer = $('mission-timer');
    if (!timer) return;
    setInterval(() => {
        const diff = Date.now() - STATE.config.missionT0;
        const d = Math.floor(diff / 86_400_000);
        const h = Math.floor((diff % 86_400_000) / 3_600_000).toString().padStart(2, '0');
        const m = Math.floor((diff % 3_600_000)  / 60_000).toString().padStart(2, '0');
        const s = Math.floor((diff % 60_000)     / 1_000).toString().padStart(2, '0');
        timer.textContent = `MET: ${d}D ${h}:${m}:${s}`;
    }, 1000);
}

// ── Dashboard View ────────────────────────────────────────────────────────────
const Dashboard = {
    render() {
        if (!STATE.noaa.plasma.length && !STATE.noaa.mag.length) return;

        const darkLayout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor:  'rgba(255,255,255,0.02)',
            font:          { color: '#a0a0a0', size: 11 },
            margin:        { t: 30, b: 45, l: 60, r: 60 }
        };
        const xaxis = { gridcolor: 'rgba(255,255,255,0.05)', tickfont: { size: 10 } };
        const opts  = { responsive: true, displayModeBar: false };

        const pTimes = STATE.noaa.plasma.map(d => d.time);
        const mTimes = STATE.noaa.mag.map(d => d.time);

        // Plasma: density + speed (dual-y)
        Plotly.react('chart-plasma', [
            {
                x: pTimes, y: STATE.noaa.plasma.map(d => d.density),
                name: 'Density (cm⁻³)',
                line: { color: '#00d4ff', width: 2 },
                fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.07)'
            },
            {
                x: pTimes, y: STATE.noaa.plasma.map(d => d.speed),
                name: 'Speed (km/s)', yaxis: 'y2',
                line: { color: '#ff6b2b', width: 2 }
            }
        ], {
            ...darkLayout, xaxis,
            yaxis:  { title: 'Density (cm⁻³)', gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis2: { title: 'Speed (km/s)',    overlaying: 'y', side: 'right', showgrid: false },
            legend: { orientation: 'h', x: 0, y: 1.12 }
        }, opts);

        // IMF: Bx By Bz Bt
        Plotly.react('chart-mag', [
            { x: mTimes, y: STATE.noaa.mag.map(d => d.bx), name: 'Bx (nT)', line: { color: '#ff5252', width: 1.5 } },
            { x: mTimes, y: STATE.noaa.mag.map(d => d.by), name: 'By (nT)', line: { color: '#00ff88', width: 1.5 } },
            { x: mTimes, y: STATE.noaa.mag.map(d => d.bz), name: 'Bz (nT)', line: { color: '#00d4ff', width: 2.5 } },
            { x: mTimes, y: STATE.noaa.mag.map(d => d.bt), name: '|B| (nT)', line: { color: '#ffffff', dash: 'dot', width: 1 } }
        ], {
            ...darkLayout, xaxis,
            yaxis: {
                title: 'Field Strength (nT)',
                gridcolor: 'rgba(255,255,255,0.05)',
                zeroline: true, zerolinecolor: 'rgba(255,255,255,0.25)'
            },
            legend: { orientation: 'h', x: 0, y: 1.12 }
        }, opts);

        // Temperature in eV
        Plotly.react('chart-temp', [{
            x: pTimes, y: STATE.noaa.plasma.map(d => d.temp / 11605),
            name: 'Proton Temp (eV)',
            line: { color: '#ffeb3b', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(255,235,59,0.07)'
        }], {
            ...darkLayout, xaxis,
            yaxis: { title: 'Temperature (eV)', gridcolor: 'rgba(255,255,255,0.05)' }
        }, opts);

        this.updateStats();
    },

    updateStats() {
        const p = STATE.noaa.plasma;
        const m = STATE.noaa.mag;
        if (!p.length) return;

        const vals = (arr, key) => arr.map(d => d[key]).filter(v => v != null && !isNaN(v));
        const avg  = vs => vs.reduce((a, b) => a + b, 0) / vs.length;

        const dens = vals(p, 'density');
        const spds = vals(p, 'speed');
        const bzs  = vals(m, 'bz');

        setEl('stat-den-avg', fmt1(avg(dens)) + ' cm⁻³');
        setEl('stat-den-max', fmt1(Math.max(...dens)) + ' cm⁻³');
        setEl('stat-spd-avg', fmt0(avg(spds)) + ' km/s');
        setEl('stat-spd-max', fmt0(Math.max(...spds)) + ' km/s');
        setEl('stat-bz-min',  fmt1(Math.min(...bzs)) + ' nT');
    }
};

// ── History View ──────────────────────────────────────────────────────────────
const HistoryView = {
    render() {
        const h = STATE.history;
        if (!h.length) {
            setEl('history-empty', 'No historical data yet. The pipeline runs every 6 hours via GitHub Actions.');
            return;
        }
        setEl('history-empty', '');

        const darkLayout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor:  'rgba(255,255,255,0.02)',
            font:          { color: '#a0a0a0', size: 11 },
            margin:        { t: 30, b: 45, l: 60, r: 20 }
        };
        const xaxis = { gridcolor: 'rgba(255,255,255,0.05)' };
        const opts  = { responsive: true, displayModeBar: false };
        const times = h.map(r => new Date(r.time));

        Plotly.react('hist-density', [{
            x: times, y: h.map(r => r.density),
            name: 'Density (cm⁻³)', line: { color: '#00d4ff', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.07)'
        }], { ...darkLayout, xaxis, yaxis: { title: 'Proton Density (cm⁻³)' } }, opts);

        Plotly.react('hist-velocity', [{
            x: times, y: h.map(r => r.velocity),
            name: 'Speed (km/s)', line: { color: '#ff6b2b', width: 2 }
        }], { ...darkLayout, xaxis, yaxis: { title: 'Bulk Speed (km/s)' } }, opts);

        Plotly.react('hist-bz', [{
            x: times, y: h.map(r => r.Bz),
            name: 'Bz (nT)', line: { color: '#00ff88', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(0,255,136,0.07)'
        }], {
            ...darkLayout, xaxis,
            yaxis: {
                title: 'IMF Bz (nT)',
                zeroline: true, zerolinecolor: 'rgba(255,255,255,0.3)',
                gridcolor: 'rgba(255,255,255,0.05)'
            }
        }, opts);

        Plotly.react('hist-beta', [{
            x: times, y: h.map(r => r.beta),
            name: 'Plasma β', line: { color: '#ffeb3b', width: 2 }
        }], {
            ...darkLayout, xaxis,
            yaxis: {
                title: 'Plasma β (thermal/magnetic pressure ratio)',
                zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)'
            }
        }, opts);
    }
};

// ── Visualizer View ───────────────────────────────────────────────────────────
const Visualizer = {
    animating: false,

    render() {
        this.drawClockAngle();
        this.animateWind();
        this.updateKpEstimate();
    },

    drawClockAngle() {
        const canvas = $('mag-clock-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Prefer pipeline mag data; fall back to NOAA
        const sc    = STATE.pipeline?.scalars;
        const mLast = STATE.noaa.mag[STATE.noaa.mag.length - 1];
        const magArr = STATE.pipeline?.mag;
        const by = (magArr?.By?.length ? magArr.By[magArr.By.length - 1] : null) ?? mLast?.by;
        const bz = sc?.Bz_nT ?? mLast?.bz;
        if (by == null || bz == null) return;

        const W = canvas.width, H = canvas.height;
        const cx = W / 2, cy = H / 2;
        const R  = Math.min(W, H) / 2 - 35;

        ctx.clearRect(0, 0, W, H);

        // Background circles
        [1, 0.66, 0.33].forEach(f => {
            ctx.strokeStyle = `rgba(255,255,255,${f * 0.08})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, R * f, 0, Math.PI * 2);
            ctx.stroke();
        });

        // Cardinal labels
        ctx.fillStyle = 'rgba(160,160,160,0.8)';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('N (Bz+, Quiet)', cx, cy - R - 10);
        ctx.fillText('S (Bz−, Active)', cx, cy + R + 20);
        ctx.textAlign = 'right';
        ctx.fillText('W', cx - R - 6, cy + 4);
        ctx.textAlign = 'left';
        ctx.fillText('E', cx + R + 6, cy + 4);

        // IMF clock angle vector
        const angle = Math.atan2(by, bz);
        const vx = cx + Math.sin(angle) * R;
        const vy = cy - Math.cos(angle) * R;

        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(vx, vy);
        ctx.stroke();

        ctx.fillStyle = '#00ff88';
        ctx.beginPath();
        ctx.arc(vx, vy, 6, 0, Math.PI * 2);
        ctx.fill();

        // Angle text in centre
        const angleDeg = sc?.clock_angle_deg ??
            ((Math.atan2(by, bz) * 180 / Math.PI + 360) % 360);
        ctx.fillStyle  = '#e0e0e0';
        ctx.font       = '13px JetBrains Mono, monospace';
        ctx.textAlign  = 'center';
        ctx.fillText(`${angleDeg.toFixed(0)}°`, cx, cy + 5);
    },

    animateWind() {
        const container = document.querySelector('#visualizer-view .wind-canvas-container');
        if (!container || this.animating) return;
        this.animating = true;

        const canvas = document.createElement('canvas');
        canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;';
        container.style.position = 'relative';
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d');

        let W, H;
        const resize = () => {
            W = canvas.width  = container.offsetWidth;
            H = canvas.height = container.offsetHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        const speedScale = (STATE.noaa.plasma[STATE.noaa.plasma.length - 1]?.speed || 400) / 100;
        const particles  = Array.from({ length: 70 }, () => ({
            x: Math.random() * 1200,
            y: Math.random() * 400,
            v: speedScale * (0.7 + Math.random() * 0.6),
            o: 0.1 + Math.random() * 0.5,
            r: 1 + Math.random() * 1.5
        }));

        const draw = () => {
            if (STATE.currentView !== 'visualizer') {
                this.animating = false;
                canvas.remove();
                return;
            }
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = '#00d4ff';
            particles.forEach(p => {
                p.x += p.v;
                if (p.x > W + 10) p.x = -10;
                ctx.globalAlpha = p.o;
                ctx.beginPath();
                ctx.arc(p.x, p.y % H, p.r, 0, Math.PI * 2);
                ctx.fill();
            });
            requestAnimationFrame(draw);
        };
        draw();
    },

    updateKpEstimate() {
        const bz = STATE.pipeline?.scalars?.Bz_nT
            ?? STATE.noaa.mag.slice(-1)[0]?.bz
            ?? 0;
        const kp = bz < -20 ? 9 : bz < -15 ? 8 : bz < -10 ? 7
                 : bz < -7  ? 6 : bz < -5  ? 5 : bz < -3  ? 4
                 : bz < -1  ? 3 : bz < 0   ? 2 : 1;
        setEl('kp-estimate', kp);
        const kpEl = $('kp-estimate');
        if (kpEl) {
            kpEl.style.color = kp >= 5 ? '#ff5252' : kp >= 3 ? '#ffeb3b' : '#00ff88';
        }
    }
};

// ── Analysis Lab View ─────────────────────────────────────────────────────────
const Lab = {
    calcSetup: false,

    render() {
        this.computeSpectral();
        if (!this.calcSetup) {
            this.setupCalculators();
            this.calcSetup = true;
        }
        this.prefillCalculators();
        this.renderEvents();
    },

    computeSpectral() {
        const psd = STATE.pipeline?.psd;

        if (psd?.valid && psd.freqs?.length) {
            Plotly.react('chart-psd', [{
                x: psd.freqs, y: psd.psd,
                type: 'scatter', mode: 'lines',
                name: 'PSD', line: { color: '#00ff88', width: 2 }
            }], {
                paper_bgcolor: 'transparent',
                plot_bgcolor:  'rgba(255,255,255,0.02)',
                font:    { color: '#a0a0a0' },
                xaxis:   { type: 'log', title: 'Frequency (Hz)',              gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis:   { type: 'log', title: 'Power Spectral Density (nT²/Hz)', gridcolor: 'rgba(255,255,255,0.05)' },
                margin:  { t: 30, b: 55, l: 75, r: 20 }
            }, { responsive: true, displayModeBar: false });
            setEl('psd-interpretation', psd.interpretation || '');
        } else {
            // Fallback: synthetic 1/f noise for illustration
            const freqs = [], power = [];
            for (let k = 1; k < 100; k++) {
                freqs.push(k / 1000);
                power.push((Math.pow(10, -2 + (Math.random() - 0.5) * 0.4)) / k);
            }
            Plotly.react('chart-psd', [{
                x: freqs, y: power,
                type: 'scatter', mode: 'lines',
                name: 'PSD (simulated)', line: { color: '#00ff88' }
            }], {
                paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(255,255,255,0.02)',
                font: { color: '#a0a0a0' },
                xaxis: { type: 'log', title: 'Frequency (Hz)' },
                yaxis: { type: 'log', title: 'Power Density' },
                margin: { t: 30, b: 55, l: 75, r: 20 }
            }, { responsive: true, displayModeBar: false });
            setEl('psd-interpretation', 'Showing illustrative 1/f spectrum — real PSD will appear when pipeline data is available.');
        }
    },

    setupCalculators() {
        document.querySelectorAll('.lab-input').forEach(i => i.addEventListener('input', () => Lab.calcAlfven()));
        this.calcAlfven();
    },

    prefillCalculators() {
        const sc = STATE.pipeline?.scalars;
        if (!sc) return;
        const el_n = $('in-n'), el_b = $('in-b');
        if (el_n) el_n.value = (+sc.density_cm3).toFixed(1);
        if (el_b) el_b.value = (+sc.B_mag_nT).toFixed(1);
        this.calcAlfven();
    },

    calcAlfven() {
        const n  = parseFloat($('in-n')?.value) || 5;
        const b  = parseFloat($('in-b')?.value) || 5;
        // Va = B / sqrt(μ₀ · ρ),  ρ = n · mp
        const va = (b * 1e-9) / Math.sqrt(4 * Math.PI * 1e-7 * n * 1e6 * 1.67e-27) / 1000;
        setEl('out-va', va.toFixed(1) + ' km/s');
    },

    renderEvents() {
        const events = STATE.pipeline?.events || [];
        const el     = $('event-log');
        if (!el) return;
        const regime = STATE.pipeline?.sw_regime || 'Unknown';
        if (!events.length) {
            el.innerHTML = `> No events detected in current observation window.\n> Solar wind conditions: <strong style="color:#00d4ff">${regime}</strong>`;
        } else {
            el.innerHTML = events
                .map(e => `> [<strong>${(e.type || '').toUpperCase()}</strong>] ${e.description || ''}`)
                .join('\n');
        }
    }
};

// ── Accordion ─────────────────────────────────────────────────────────────────
function setupAccordions() {
    document.querySelectorAll('.accordion-header').forEach(h => {
        h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
    });
}

// ── Advanced Raw-JSON Toggle ──────────────────────────────────────────────────
function setupAdvancedToggle() {
    const btn   = $('toggle-advanced');
    const panel = $('advanced-panel');
    if (!btn || !panel) return;

    btn.addEventListener('click', () => {
        const isOpen = panel.style.display === 'block';
        panel.style.display = isOpen ? 'none' : 'block';
        btn.textContent     = isOpen ? '⚙ Show Raw Data (JSON)' : '✕ Hide Raw Data';
        if (!isOpen && STATE.pipeline) {
            $('raw-json-content').textContent = JSON.stringify(STATE.pipeline, null, 2);
        }
    });
}

// ── Mobile Menu Toggle ────────────────────────────────────────────────────────
function setupMobileMenu() {
    const toggle = $('mobile-menu-toggle');
    const links  = document.querySelector('.nav-links');
    if (!toggle || !links) return;
    toggle.addEventListener('click', () => links.classList.toggle('open'));
}

// ── CSV Download ──────────────────────────────────────────────────────────────
function setupDownload() {
    const btn = $('btn-download-csv');
    if (!btn) return;
    btn.addEventListener('click', () => {
        if (!STATE.noaa.plasma.length) return;
        const rows = [['time_utc', 'density_cm3', 'speed_kms', 'temp_K', 'bx_nT', 'by_nT', 'bz_nT', 'bt_nT']];
        const magMap = {};
        STATE.noaa.mag.forEach(m => { magMap[m.time.toISOString()] = m; });
        STATE.noaa.plasma.forEach(p => {
            const m = magMap[p.time.toISOString()] || {};
            rows.push([p.time.toISOString(), p.density ?? '', p.speed ?? '', p.temp ?? '',
                       m.bx ?? '', m.by ?? '', m.bz ?? '', m.bt ?? '']);
        });
        const csv = rows.map(r => r.join(',')).join('\n');
        const a   = document.createElement('a');
        a.href     = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
        a.download = 'solar-wind-' + new Date().toISOString().slice(0, 10) + '.csv';
        a.click();
    });
}

// ── Initialisation ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Global click delegation for [data-view] links
    document.body.addEventListener('click', e => {
        const t = e.target.closest('[data-view]');
        if (t) switchView(t.dataset.view);
    });

    startMissionTimer();
    setupAccordions();
    setupAdvancedToggle();
    setupMobileMenu();
    setupDownload();

    fetchAll();
});
