// --- NOVA ADITYA-L1 GLOBAL ENGINE ---

// Constants
const MISSION_T0 = new Date("2023-09-02T11:50:00+05:30").getTime(); // Launch: Sept 2, 2023
const LATEST_URL = 'data/latest.json';

// Global State
window.adityaTelemetry = null;

// Register GSAP Plugins
if (typeof gsap !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);
}

// --- Mission Timer (MET) ---
function startMissionTimer() {
    const metElement = document.getElementById("mission-timer");
    if (!metElement) return;

    setInterval(() => {
        const now = Date.now();
        const diff = now - MISSION_T0;

        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const secs = Math.floor((diff % (1000 * 60)) / 1000);

        metElement.innerText = `MET: ${days}D ${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }, 1000);
}

// --- Shared Components Injection ---
function initSharedComponents() {
    // 1. Navigation & HUD Logic
    const nav = document.querySelector('nav.nova-nav');
    if (nav) {
        const currentPath = window.location.pathname.split('/').pop() || 'index.html';
        nav.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) link.classList.add('active');
        });
    }

    // 2. Preloader Logic
    const preloader = document.getElementById("preloader");
    if (preloader) {
        const tl = gsap.timeline({
            onComplete: () => {
                gsap.to("#preloader", {
                    opacity: 0, duration: 1,
                    onComplete: () => {
                        preloader.style.display = "none";
                        if (typeof initHeroAnimations === 'function') initHeroAnimations();
                    }
                });
            }
        });
        tl.to(".preloader-bar-fill", { width: "100%", duration: 2, ease: "power2.inOut" });
        tl.to(".preloader-status", { text: "MISSION READY", duration: 0.5 }, "-=0.2");
    }

    startMissionTimer();
}

// --- Global Telemetry Sync ---
async function syncGlobalTelemetry() {
    try {
        const res = await fetch(LATEST_URL + '?t=' + Date.now());
        const data = await res.json();
        window.adityaTelemetry = data;

        // Update Universal HUD Mini-Metrics
        const hudDensity = document.getElementById('hud-density');
        const hudSpeed = document.getElementById('hud-speed');
        const hudMag = document.getElementById('hud-mag');

        if (hudDensity && data.scalars.density_cm3) hudDensity.innerText = data.scalars.density_cm3.toFixed(1);
        if (hudSpeed && data.scalars.velocity_kms) hudSpeed.innerText = data.scalars.velocity_kms.toFixed(0);
        if (hudMag && data.scalars.B_mag_nT) hudMag.innerText = data.scalars.B_mag_nT.toFixed(1);

        // Broadcast to specific page controllers (dashboard, lab)
        document.dispatchEvent(new CustomEvent('adityaDataUpdate', { detail: data }));

    } catch (e) {
        console.warn("Telemetry offline. Using fallback simulation.");
    }
}

// --- Scroll & Reveal Logic ---
function initScrollAnimations() {
    gsap.utils.toArray(".reveal-text").forEach(text => {
        gsap.to(text, {
            scrollTrigger: { trigger: text, start: "top 85%", toggleActions: "play none none none" },
            y: 0, opacity: 1, duration: 1, ease: "power3.out"
        });
    });

    gsap.utils.toArray(".sun-parallax").forEach(bg => {
        gsap.to(bg, {
            scrollTrigger: { trigger: bg.parentElement, start: "top top", end: "bottom top", scrub: true },
            y: 150, ease: "none"
        });
    });
}

// --- Particle Background ---
function initParticles() {
    const canvas = document.getElementById("solar-wind-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W, H;
    let particles = [];
    function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
    class Particle {
        constructor() { this.reset(); }
        reset() {
            this.x = Math.random() * W; this.y = Math.random() * H;
            this.r = Math.random() * 2 + 0.5; this.speed = Math.random() * 2 + 0.5;
            this.alpha = Math.random() * 0.4 + 0.1;
        }
        update() { this.x += this.speed; if (this.x > W) { this.x = -10; this.y = Math.random() * H; } }
    }
    function animate() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            p.update();
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 107, 26, ${p.alpha})`;
            ctx.fill();
        });
        requestAnimationFrame(animate);
    }
    resize();
    for (let i = 0; i < 80; i++) particles.push(new Particle());
    animate();
    window.addEventListener("resize", resize);
}

// --- Entry Point ---
document.addEventListener("DOMContentLoaded", () => {
    initSharedComponents();
    initScrollAnimations();
    initParticles();
    syncGlobalTelemetry();
    setInterval(syncGlobalTelemetry, 300000); // 5 min global refresh
});
