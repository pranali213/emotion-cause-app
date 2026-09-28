/* ── Index page ── */

// Apply CSS custom property --c from data-color on emo-wheel items
document.querySelectorAll(".emo-wheel-item[data-color]").forEach(el => {
  el.style.setProperty("--c", el.dataset.color);
});

const EMO_COLORS = { joy:"#F59E0B", sadness:"#3B82F6", anger:"#EF4444", fear:"#8B5CF6", surprise:"#10B981", disgust:"#6B7280", neutral:"#94A3B8" };
const EMO_ICONS  = { joy:"😊", sadness:"😢", anger:"😠", fear:"😨", surprise:"😲", disgust:"🤢", neutral:"😐" };

// ── New Conversation Modal ────────────────────────────────────
["btnNewConv","emptyNewBtn"].forEach(id => document.getElementById(id)?.addEventListener("click", () => openModal("modalNewConv")));
["closeModalNewConv","cancelNewConv"].forEach(id => document.getElementById(id)?.addEventListener("click", () => closeModal("modalNewConv")));
document.getElementById("modalNewConv")?.addEventListener("click", e => { if (e.target.id==="modalNewConv") closeModal("modalNewConv"); });
document.getElementById("newConvTitle")?.addEventListener("keydown", e => { if (e.key==="Enter") document.getElementById("confirmNewConv").click(); });

document.getElementById("confirmNewConv")?.addEventListener("click", async () => {
  const title = document.getElementById("newConvTitle").value.trim() || "Untitled Conversation";
  const btn = document.getElementById("confirmNewConv");
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Creating…';
  try {
    const r = await fetch("/api/conversations", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({title}) });
    const c = await r.json();
    window.location.href = `/conversation/${c.id}`;
  } catch { toast("Failed to create conversation", "error"); btn.disabled=false; btn.textContent="Create & Open"; }
});

// ── Load Samples ──────────────────────────────────────────────
["btnLoadSamples","emptyLoadBtn"].forEach(id => {
  document.getElementById(id)?.addEventListener("click", async () => {
    if (!confirm("Load sample conversations? This adds 8 example dialogues.")) return;
    const btn = document.getElementById(id);
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Loading…';
    try {
      const r = await fetch("/api/load-samples", { method:"POST" });
      const d = await r.json();
      if (d.error) { toast(d.error, "error"); return; }
      toast(`Loaded ${d.loaded} sample conversations!`, "success");
      setTimeout(() => location.reload(), 800);
    } catch { toast("Failed to load samples", "error"); }
    finally { btn.disabled=false; btn.textContent = id==="btnLoadSamples"?"📂 Load Samples":"📂 Load Samples"; }
  });
});

// ── Delete conversation ───────────────────────────────────────
document.querySelectorAll(".del-conv-btn").forEach(btn => {
  btn.addEventListener("click", async e => {
    e.preventDefault(); e.stopPropagation();
    const id = btn.dataset.id;
    if (!confirm("Delete this conversation and all its data?")) return;
    await fetch(`/api/conversations/${id}`, { method:"DELETE" });
    const card = btn.closest(".conv-card");
    card.style.transition = "all .2s ease"; card.style.opacity="0"; card.style.transform="scale(.9)";
    setTimeout(() => {
      card.remove();
      if (!document.querySelector(".conv-card")) location.reload();
    }, 210);
    toast("Deleted", "success");
  });
});

// ── Search conversations ──────────────────────────────────────
document.getElementById("searchConv")?.addEventListener("input", function() {
  const q = this.value.toLowerCase();
  document.querySelectorAll(".conv-card").forEach(c => {
    c.style.display = c.dataset.title.toLowerCase().includes(q) ? "" : "none";
  });
});

// ── Quick Predict (all 4 models) ──────────────────────────────
const qpText = document.getElementById("qpText");
const qpBtn  = document.getElementById("qpBtn");
const qpRes  = document.getElementById("qpResults");

async function runQuickPredict() {
  const text = qpText.value.trim();
  if (!text) { toast("Enter some text first", "error"); return; }
  qpBtn.disabled=true; qpBtn.innerHTML='<span class="spinner"></span>';
  qpRes.classList.add("hidden");
  try {
    const r = await fetch("/api/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({text}) });
    const d = await r.json();
    let html = "";
    for (const [model, pred] of Object.entries(d.per_model)) {
      const col = EMO_COLORS[pred.emotion] || "#94a3b8";
      html += `<div class="qp-model-card" style="border-color:${col}30">
        <div class="qp-model-name">${esc(model)}</div>
        <span class="qp-emo-icon">${pred.icon}</span>
        <div class="qp-emo-label" style="color:${col}">${cap(pred.emotion)}</div>
        <div class="qp-conf">${pct(pred.confidence)}% confidence</div>
      </div>`;
    }
    const ec = EMO_COLORS[d.ensemble]||"#94a3b8";
    html += `<div class="qp-ensemble">
      <div class="qp-ensemble-label">🗳 Ensemble Vote</div>
      <div class="qp-ensemble-val" style="color:${ec}">${EMO_ICONS[d.ensemble]||"😐"} ${cap(d.ensemble)}</div>
    </div>`;
    qpRes.innerHTML = html;
    qpRes.classList.remove("hidden");
  } catch { toast("Prediction failed","error"); }
  finally { qpBtn.disabled=false; qpBtn.textContent="Predict"; }
}
qpBtn?.addEventListener("click", runQuickPredict);
qpText?.addEventListener("keydown", e => { if (e.key==="Enter" && !e.shiftKey) { e.preventDefault(); runQuickPredict(); } });

// ── Quick Analyse ──────────────────────────────────────────────
const qaUtts  = [];
const qaList  = document.getElementById("qaList");
const qaSpeaker=document.getElementById("qaSpeaker");
const qaTextEl = document.getElementById("qaText");
const qaModel  = document.getElementById("qaModel");
const qaRes    = document.getElementById("qaResults");

function renderQaList() {
  qaList.innerHTML = "";
  qaUtts.forEach((u, i) => {
    const el = document.createElement("div");
    el.className = "qa-utt";
    el.innerHTML = `<span class="qa-speaker">${esc(u.speaker)}</span><span class="qa-text">${esc(u.text)}</span><button class="qa-del" data-i="${i}">✕</button>`;
    qaList.appendChild(el);
  });
  qaList.querySelectorAll(".qa-del").forEach(b => b.addEventListener("click", () => { qaUtts.splice(+b.dataset.i,1); renderQaList(); }));
}

function addQaUtt() {
  const speaker = (qaSpeaker.value||"Speaker A").trim();
  const text    = qaTextEl.value.trim();
  if (!text) { toast("Enter text","error"); return; }
  qaUtts.push({ speaker, text });
  qaTextEl.value = "";
  renderQaList();
  qaTextEl.focus();
  // alternate speaker
  if (qaUtts.length >= 1) {
    const last  = qaUtts[qaUtts.length-1].speaker;
    const all   = [...new Set(qaUtts.map(u=>u.speaker))];
    qaSpeaker.value = all.length===1 ? (last==="Alice"?"Bob":"Alice") : (all.find(s=>s!==last)||all[0]);
  }
}

document.getElementById("qaAddBtn")?.addEventListener("click", addQaUtt);
qaTextEl?.addEventListener("keydown", e => { if (e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); addQaUtt(); } });
document.getElementById("qaClearBtn")?.addEventListener("click", () => { qaUtts.length=0; renderQaList(); qaRes.innerHTML=""; qaRes.classList.add("hidden"); });

document.getElementById("qaAnalyseBtn")?.addEventListener("click", async () => {
  if (qaUtts.length < 2) { toast("Add at least 2 turns","error"); return; }
  const btn = document.getElementById("qaAnalyseBtn");
  btn.disabled=true; btn.innerHTML='<span class="spinner"></span> Analysing…';
  qaRes.classList.add("hidden");
  try {
    const r = await fetch("/api/quick-analyse", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ utterances: qaUtts, model: qaModel.value })
    });
    const d = await r.json();
    qaRes.innerHTML = buildResultsHTML(d);
    qaRes.classList.remove("hidden");
    toast(`Found ${d.pairs.length} EC pair(s)`, "success");
  } catch { toast("Analysis failed","error"); }
  finally { btn.disabled=false; btn.textContent="🤖 Run Analysis"; }
});

function buildResultsHTML(data) {
  const s = data.stats;
  let html = `<div class="stats-strip" style="border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden;margin-bottom:12px">
    <div class="stat-box"><span class="stat-num">${s.total_utterances}</span><span class="stat-label">Turns</span></div>
    <div class="stat-box"><span class="stat-num">${s.total_pairs}</span><span class="stat-label">EC Pairs</span></div>
    <div class="stat-box"><span class="stat-num">${cap(s.dominant_emotion)}</span><span class="stat-label">Dominant</span></div>
    <div class="stat-box"><span class="stat-num">${esc(s.model_used.split(" ")[0])}</span><span class="stat-label">Model</span></div>
  </div>`;

  if (!data.pairs.length) {
    html += `<p class="muted-text" style="text-align:center;padding:16px">No emotion-cause pairs detected. Try more expressive sentences.</p>`;
  } else {
    data.pairs.forEach(p => {
      const eu = data.utterances.find(u => u.turn_index===p.emotion_turn);
      const cu = data.utterances.find(u => u.turn_index===p.cause_turn);
      const col = p.color||"#94a3b8";
      html += `<div class="pair-card">
        <div class="pair-header">
          <span class="pair-emo-badge" style="background:${col}18;color:${col};border-color:${col}35">${p.icon} ${cap(p.emotion)}</span>
          <span class="conf-bar-wrap"><span class="conf-bar-fill" style="width:${pct(p.confidence)}%;background:${col}"></span></span>
          <span class="pair-conf-text">${pct(p.confidence)}%</span>
        </div>
        <div class="pair-body">
          <div class="pair-block emo-block"><span class="pb-label">😤 Emotion · T${p.emotion_turn+1} · ${esc(eu?.speaker||"")}</span><p class="pb-text">${esc(eu?.text||"")}</p></div>
          <div class="pair-arrow">↑ caused by</div>
          <div class="pair-block cause-block"><span class="pb-label">💡 Cause · T${p.cause_turn+1} · ${esc(cu?.speaker||"")}</span><p class="pb-text">${esc(cu?.text||"")}</p>${p.cause_span?`<span class="cause-span-tag">"${esc(p.cause_span)}"</span>`:""}</div>
        </div>
        <p class="pair-explain">ℹ ${esc(p.explanation)}</p>
      </div>`;
    });
  }
  return html;
}
