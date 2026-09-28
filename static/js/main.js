/* ── Shared utilities ── */
window.toast = function(msg, type = "info", ms = 3500) {
  const c = document.getElementById("toastContainer");
  if (!c) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => {
    el.style.transition = "all .25s ease";
    el.style.opacity = "0";
    el.style.transform = "translateX(110%)";
    setTimeout(() => el.remove(), 260);
  }, ms);
};

window.esc = function(str) {
  const d = document.createElement("div");
  d.textContent = String(str ?? "");
  return d.innerHTML;
};

window.cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;

window.pct = n => Math.round((n || 0) * 100);

// Sidebar toggle
const sidebar   = document.getElementById("sidebar");
const layoutMain= document.getElementById("layoutMain");
const hamburger = document.getElementById("hamburger");

function toggleSidebar() {
  if (window.innerWidth <= 700) {
    sidebar.classList.toggle("open");
  } else {
    sidebar.classList.toggle("collapsed");
    layoutMain.classList.toggle("expanded");
  }
}
hamburger?.addEventListener("click", toggleSidebar);
document.addEventListener("click", e => {
  if (window.innerWidth <= 700 && sidebar?.classList.contains("open"))
    if (!sidebar.contains(e.target) && e.target !== hamburger)
      sidebar.classList.remove("open");
});

// Modal helpers
window.openModal  = id => document.getElementById(id)?.classList.remove("hidden");
window.closeModal = id => document.getElementById(id)?.classList.add("hidden");

// Hydrate mini-bar fills rendered with data-width / data-color (avoids Jinja CSS linter errors)
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".mini-fill[data-width]").forEach(el => {
    el.style.width = el.dataset.width + "%";
    if (el.dataset.color) el.style.background = el.dataset.color;
  });
});

// ── GA4 custom event helper ───────────────────────────────────
window.trackEvent = function(eventName, params) {
  if (typeof gtag === "function") {
    gtag("event", eventName, params || {});
  }
};
