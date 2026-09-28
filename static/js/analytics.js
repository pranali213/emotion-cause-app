/* ── Analytics page ── */

const D = window.ANALYTICS_DATA || {};
const EMO_COLORS = {
  joy:"#F59E0B", sadness:"#3B82F6", anger:"#EF4444",
  fear:"#8B5CF6", surprise:"#10B981", disgust:"#6B7280", neutral:"#94A3B8",
};

Chart.defaults.color        = "#64748b";
Chart.defaults.font.family  = "'Inter', system-ui, sans-serif";

// ── Metric Comparison Bar Chart ───────────────────────────────
(function () {
  const ctx = document.getElementById("chartMetrics");
  if (!ctx || !D.comparison) return;

  const models    = D.comparison.map(r => r.model);
  const accuracy  = D.comparison.map(r => +(r.accuracy  * 100).toFixed(1));
  const precision = D.comparison.map(r => +(r.precision * 100).toFixed(1));
  const recall    = D.comparison.map(r => +(r.recall    * 100).toFixed(1));
  const f1        = D.comparison.map(r => +(r.f1        * 100).toFixed(1));

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: models,
      datasets: [
        { label: "Accuracy",  data: accuracy,  backgroundColor: "rgba(99,102,241,.75)", borderRadius: 4 },
        { label: "Precision", data: precision, backgroundColor: "rgba(16,185,129,.75)", borderRadius: 4 },
        { label: "Recall",    data: recall,    backgroundColor: "rgba(245,158,11,.75)", borderRadius: 4 },
        { label: "F1 Score",  data: f1,        backgroundColor: "rgba(239,68,68,.75)",  borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#252d47" } },
        y: { min: 0, max: 100,
             ticks: { color: "#94a3b8", callback: v => v + "%" },
             grid:  { color: "#252d47" } },
      },
    },
  });
})();

// ── Emotion Distribution Donut ────────────────────────────────


// ── Refresh button ────────────────────────────────────────────
document.getElementById("btnRefresh")?.addEventListener("click", () => location.reload());
