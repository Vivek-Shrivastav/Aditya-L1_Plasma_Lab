// --- NOVA Dashboard Controller ---

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

function updateDashboardUI(data) {
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

    renderCharts(data);
}

function renderCharts(data) {
    const mom = data.moments;
    if (!mom || !document.getElementById('chart-overview')) return;

    const times = mom.times.map(t => new Date(t));

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
    
    Plotly.react('chart-overview', traces, layout, { responsive: true, displayModeBar: false });
}

// Listen for global telemetry updates from nova.js
document.addEventListener('adityaDataUpdate', (e) => {
    updateDashboardUI(e.detail);
});

document.addEventListener('DOMContentLoaded', () => {
    // Initial UI check if data already exists
    if (window.adityaTelemetry) updateDashboardUI(window.adityaTelemetry);
});
