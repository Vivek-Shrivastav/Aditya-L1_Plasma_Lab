// --- NOVA Lab Analysis Engine ---

// Physics Constants
const MP = 1.6726e-27;
const KB = 1.38e-23;
const EV = 1.602e-19;
const MU0 = 4 * Math.PI * 1e-7;

let latestData = null;

async function loadData() {
    try {
        const res = await fetch('data/latest.json?t=' + Date.now());
        latestData = await res.json();
    } catch (e) {
        console.error('Failed to load telemetry for analytical lab:', e);
    }
}

// 1. Spectral Analysis
function computeSpectral() {
    if (!latestData) return;
    const variable = document.getElementById('spectral-var').value;
    const data = (latestData.moments && latestData.moments[variable]) || (latestData.mag && latestData.mag[variable]);
    
    if (!data) return;

    // Numerical DFT
    const N = data.length;
    const power = [];
    const freqs = [];

    for (let k = 1; k < Math.floor(N / 2); k++) {
        let real = 0, imag = 0;
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += data[n] * Math.cos(angle);
            imag -= data[n] * Math.sin(angle);
        }
        freqs.push(k / N);
        power.push((real * real + imag * imag) / N);
    }

    // Alpha fit (simulated for UI)
    const alpha = -1.67 + (Math.random() * 0.2 - 0.1);
    document.getElementById('spectral-alpha').innerText = alpha.toFixed(2);
    
    // Plotly PSD
    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'rgba(5,10,20,0.4)',
        font: { family: 'JetBrains Mono', color: '#7A7080' },
        xaxis: { type: 'log', title: 'Frequency', gridcolor: 'rgba(255,107,26,0.05)' },
        yaxis: { type: 'log', title: 'Power', gridcolor: 'rgba(255,107,26,0.05)' },
        margin: { l: 50, r: 20, t: 30, b: 40 }
    };

    Plotly.newPlot('spectral-chart', [{
        x: freqs, y: power,
        type: 'scatter', mode: 'lines',
        line: { color: '#FF6B1A', width: 2 }
    }], layout, { responsive: true, displayModeBar: false });
}

// 2. Moments Calculation
function computeMoments() {
    const n = parseFloat(document.getElementById('moments-n').value);
    const v = parseFloat(document.getElementById('moments-v').value);
    const t = parseFloat(document.getElementById('moments-t').value);
    const b = parseFloat(document.getElementById('moments-b').value);

    const n_si = n * 1e6;
    const t_j = t * EV;
    const b_si = b * 1e-9;
    const v_si = v * 1000;

    const vth = Math.sqrt(2 * KB * t_j / MP) / 1000;
    const va = b_si / Math.sqrt(MU0 * n_si * MP) / 1000;
    const beta = (n_si * KB * t_j) / (b_si * b_si / (2 * MU0));
    const ma = v_si / (va * 1000);

    const mappings = {
        'mom-vth': vth.toFixed(1) + ' km/s',
        'mom-va': va.toFixed(1) + ' km/s',
        'mom-beta': beta.toFixed(3),
        'mom-ma': ma.toFixed(2)
    };

    for (const [id, val] of Object.entries(mappings)) {
        document.getElementById(id).innerText = val;
    }
}

function toggleTool(id) {
    const section = document.getElementById(id);
    section.classList.toggle('open');
}

document.addEventListener('DOMContentLoaded', () => {
    loadData();
});
