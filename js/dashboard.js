// --- NOVA Dashboard Controller ---

const LATEST_URL = 'data/latest.json';
const HISTORY_URL = 'data/history.json';

const LAYOUT_BASE = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(5,10,20,0.6)',
    font: { family: 'JetBrains Mono, monospace', color: '#7A7080', size: 10 },
    xaxis: {
        gridcolor: 'rgba(255,107,26,0.05)',
        linecolor: 'rgba(255,107,26,0.1)',
        tickfont: { color: '#7A7080' }
    },
    yaxis: {
        gridcolor: 'rgba(255,107,26,0.05)',
        linecolor: 'rgba(255,107,26,0.1)',
        tickfont: { color: '#7A7080' }
    },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    showlegend: true,
    legend: { x: 0, y: 1.1, orientation: 'h', font: { size: 9 } }
};

let latestData = null;

async function syncData() {
    const btn = document.getElementById('sync-btn');
    if (btn) btn.innerText = 'SYNCING...';

    try {
        const res = await fetch(LATEST_URL + '?t=' + Date.now());
        latestData = await res.json();
        
        updateHUD(latestData);
        renderCharts(latestData);
        
        if (btn) btn.innerText = 'SYNC COMPLETE';
        setTimeout(() => { if (btn) btn.innerText = 'SYNC NOMINAL'; }, 2000);
    } catch (e) {
        console.error("Sync failed", e);
        if (btn) btn.innerText = 'SYNC FAILED';
    }
}

function updateHUD(data) {
    const s = data.scalars;
    if (!s) return;

    const mappings = {
        'val-density': s.density_cm3,
        'val-speed': s.velocity_kms,
        'val-temp': s.temperature_eV,
        'val-mag': s.B_mag_nT
    };

    for (const [id, val] of Object.entries(mappings)) {
        const el = document.getElementById(id);
        if (el) {
            el.innerText = val.toFixed(id.includes('speed') ? 0 : 2);
            el.setAttribute('data-value', val);
        }
    }
}

function renderCharts(data) {
    const mom = data.moments;
    if (!mom) return;

    const times = mom.times.map(t => new Date(t));

    // Overview Chart
    const traces = [
        {
            x: times, y: mom.density,
            name: 'Density', line: { color: '#FF6B1A', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(255, 107, 26, 0.05)'
        },
        {
            x: times, y: mom.velocity,
            name: 'Speed', yaxis: 'y2', line: { color: '#FFD166', width: 2 }
        }
    ];

    const layout = { ...LAYOUT_BASE };
    layout.yaxis2 = { overlaying: 'y', side: 'right', showgrid: false };
    
    Plotly.newPlot('chart-overview', traces, layout, { responsive: true, displayModeBar: false });
}

document.addEventListener('DOMContentLoaded', () => {
    syncData();
    setInterval(syncData, 600000); // 10 min refresh
});
