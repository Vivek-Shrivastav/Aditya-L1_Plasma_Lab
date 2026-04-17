// --- NOVA Aditya-L1 Engine ---

// Register GSAP Plugins
if (typeof gsap !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);
}

// --- Shared Components ---
function initSharedComponents() {
    // 1. Navigation Injection (if not present)
    const nav = document.querySelector('nav.nova-nav');
    if (nav) {
        const currentPath = window.location.pathname.split('/').pop() || 'index.html';
        nav.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }

    // 2. Preloader Logic
    const preloader = document.getElementById("preloader");
    if (preloader) {
        const tl = gsap.timeline({
            onComplete: () => {
                gsap.to("#preloader", {
                    opacity: 0,
                    duration: 1,
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
}

// --- Scroll Animations ---
function initScrollAnimations() {
    // Reveal sections
    gsap.utils.toArray(".reveal-text").forEach(text => {
        gsap.to(text, {
            scrollTrigger: {
                trigger: text,
                start: "top 85%",
                toggleActions: "play none none none"
            },
            y: 0,
            opacity: 1,
            duration: 1,
            ease: "power3.out"
        });
    });

    // Parallax background (General)
    gsap.utils.toArray(".sun-parallax").forEach(bg => {
        gsap.to(bg, {
            scrollTrigger: {
                trigger: bg.parentElement,
                start: "top top",
                end: "bottom top",
                scrub: true
            },
            y: 150,
            ease: "none"
        });
    });

    // Count Up HUD Values
    gsap.utils.toArray(".hud-value").forEach(val => {
        const targetValue = parseFloat(val.getAttribute('data-value') || val.innerText);
        if (isNaN(targetValue)) return;

        gsap.from(val, {
            scrollTrigger: {
                trigger: val,
                start: "top 90%"
            },
            innerText: 0,
            duration: 1.5,
            snap: { innerText: 0.1 },
            ease: "power1.out"
        });
    });
}

// --- Solar Wind Particle System ---
function initParticles() {
    const canvas = document.getElementById("solar-wind-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let W, H;
    let particles = [];

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }

    class Particle {
        constructor() { this.reset(); }
        reset() {
            this.x = Math.random() * W;
            this.y = Math.random() * H;
            this.r = Math.random() * 2 + 0.5;
            this.speed = Math.random() * 2 + 0.5;
            this.alpha = Math.random() * 0.4 + 0.1;
        }
        update() {
            this.x += this.speed;
            if (this.x > W) { this.x = -10; this.y = Math.random() * H; }
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 107, 26, ${this.alpha})`;
            ctx.fill();
        }
    }

    function animate() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            p.update();
            // Inlined draw for performance
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
});

// --- Ask SURYA Logic ---
function initAskSurya() {
    const input = document.getElementById("surya-input");
    const submit = document.getElementById("surya-submit");
    const output = document.getElementById("surya-output");

    if (!submit) return;

    submit.addEventListener("click", async () => {
        const query = input.value.trim();
        if (!query) return;

        output.innerHTML = `<div class="typing-indicator">Surya is thinking...</div>`;
        
        // Simulated AI response based on mission knowledge
        setTimeout(() => {
            let response = "I am processing the latest telemetry from Aditya-L1. ";
            if (query.toLowerCase().includes("solar wind")) {
                response += "The solar wind is a stream of charged particles released from the upper atmosphere of the Sun, called the corona. Aditya-L1's SWIS and STEPS payloads track this constantly.";
            } else if (query.toLowerCase().includes("l1")) {
                response += "The L1 point is a gravitational 'sweet spot' 1.5 million km from Earth where the Sun's and Earth's gravity balance out, allowing for an uninterrupted view of the Sun.";
            } else {
                response += "That's a fascinating aspect of solar physics. Aditya-L1 is designed specifically to study these phenomena in the corona and interplanetary space.";
            }
            output.innerHTML = `<div class="surya-response">${response}</div>`;
        }, 1500);
    });
}

// --- State Management & Live Data ---
async function fetchLatestData() {
    try {
        const res = await fetch("data/latest.json?t=" + Date.now());
        const data = await res.json();
        const s = data.scalars || {};

        if (document.getElementById("val-density")) document.getElementById("val-density").innerText = s.density_cm3 || "3.8";
        if (document.getElementById("val-speed")) document.getElementById("val-speed").innerText = s.velocity_kms || "412";
        if (document.getElementById("val-temp")) document.getElementById("val-temp").innerText = s.temperature_eV || "6.8";
        if (document.getElementById("val-mag")) document.getElementById("val-mag").innerText = s.B_mag_nT || "5.0";

    } catch (e) {
        console.warn("Using simulation fallback data");
    }
}

// --- Entry Point ---
document.addEventListener("DOMContentLoaded", () => {
    initPreloader();
    initScrollAnimations();
    initParticles();
    initAskSurya();
    fetchLatestData();
});
