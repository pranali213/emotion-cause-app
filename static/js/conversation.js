/* ── Conversation page ── */

const CONV_ID = window.CONV_ID;
const EMO_COLORS = { joy:"#F59E0B",sadness:"#3B82F6",anger:"#EF4444",fear:"#8B5CF6",surprise:"#10B981",disgust:"#6B7280",neutral:"#94A3B8" };
const AV_COLORS  = ["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6","#3b82f6","#ec4899","#14b8a6"];

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll(".rtab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".rtab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".rtab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab" + cap(btn.dataset.tab))?.classList.add("active");
  });
});

// ── Add utterance ─────────────────────────────────────────────
const dialogueList   = document.getElementById("dialogueList");
const dialogueScroll = document.getElementById("dialogueScroll");
const newSpeaker = document.getElementById("newSpeaker");
const newText    = document.getElementById("newText");
const btnAddUtt  = document.getElementById("btnAddUtt");
const uttCountBadge = document.getElementById("uttCountBadge");
const statTurns  = document.getElementById("statTurns");
let   turnCount  = document.querySelectorAll(".utt-row").length;

function updateCount(n) {
  turnCount = n;
  uttCountBadge.textContent = `${n} turn${n!==1?"s":""}`;
  statTurns.textContent = n;
}

function buildUttRow(u) {
  const side = u.turn_index % 2 === 0 ? "utt-left" : "utt-right";
  const avCol = AV_COLORS[u.turn_index % 8];
  const div = document.createElement("div");
  div.className = `utt-row ${side}`;
  div.dataset.id   = u.id;
  div.dataset.turn = u.turn_index;
  div.innerHTML = `
    <div class="utt-avatar" style="background:${avCol}">${esc(u.speaker[0].toUpperCase())}</div>
    <div class="utt-bubble">
      <div class="utt-meta">
        <span class="utt-speaker">${esc(u.speaker)}</span>
        <span class="utt-turn">T${u.turn_index+1}</span>
      </div>
      <p class="utt-text" id="uttText${u.id}">${esc(u.text)}</p>
      <div class="utt-footer" id="uttFooter${u.id}">${buildEmoBadge(u)}</div>
    </div>
    <div class="utt-actions">
      <button class="icon-btn edit-btn"   data-id="${u.id}" title="Edit">✏️</button>
      <button class="icon-btn danger del-btn" data-id="${u.id}" title="Delete">🗑</button>
    </div>`;
  attachUttEvents(div);
  return div;
}

function buildEmoBadge(u) {
  if (!u.emotion || u.emotion === "neutral") return "";
  const col = u.color || EMO_COLORS[u.emotion] || "#94a3b8";
  return `<span class="emo-badge" style="background:${col}18;color:${col};border-color:${col}35">${u.icon||""} ${cap(u.emotion)} <small>${pct(u.confidence)}%</small></span>`;
}

function attachUttEvents(row) {
  row.querySelector(".edit-btn")?.addEventListener("click", () => openEditModal(row));
  row.querySelector(".del-btn")?.addEventListener("click",  () => deleteUtt(row));
}
document.querySelectorAll(".utt-row").forEach(attachUttEvents);

async function addUtt() {
  const speaker = (newSpeaker.value||"Speaker A").trim();
  const text    = newText.value.trim();
  const model   = document.getElementById("modelSelect").value;
  if (!text) { toast("Enter some text","error"); return; }
  btnAddUtt.disabled=true; btnAddUtt.innerHTML='<span class="spinner"></span>';
  try {
    const r = await fetch(`/api/conversations/${CONV_ID}/utterances`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({speaker, text, model})
    });
    if (!r.ok) throw new Error();
    const u = await r.json();
    document.getElementById("emptyDialogue")?.remove();
    dialogueList.appendChild(buildUttRow(u));
    dialogueScroll.scrollTop = dialogueScroll.scrollHeight;
    newText.value = "";
    updateCount(turnCount+1);
    // auto-alternate
    const speakers = [...new Set([...document.querySelectorAll(".utt-speaker")].map(s=>s.textContent))];
    newSpeaker.value = speakers.find(s=>s!==speaker) || (speaker==="Alice"?"Bob":"Alice");
    toast("Turn added","success");
  } catch { toast("Failed to add turn","error"); }
  finally { btnAddUtt.disabled=false; btnAddUtt.textContent="＋ Add"; }
}
btnAddUtt?.addEventListener("click", addUtt);
newText?.addEventListener("keydown", e => { if (e.key==="Enter"&&!e.shiftKey){e.preventDefault();addUtt();} });

// ── Delete utterance ──────────────────────────────────────────
async function deleteUtt(row) {
  if (!confirm("Delete this utterance?")) return;
  await fetch(`/api/utterances/${row.dataset.id}`, {method:"DELETE"});
  row.style.transition="all .18s ease"; row.style.opacity="0"; row.style.transform="scale(.92)";
  setTimeout(()=>row.remove(), 190);
  updateCount(turnCount-1);
  toast("Deleted","success");
}

// ── Edit utterance ────────────────────────────────────────────
const modalEditUtt = document.getElementById("modalEditUtt");
const editUttId    = document.getElementById("editUttId");
const editSpeaker  = document.getElementById("editSpeaker");
const editText     = document.getElementById("editText");

function openEditModal(row) {
  editUttId.value   = row.dataset.id;
  editSpeaker.value = row.querySelector(".utt-speaker").textContent;
  editText.value    = row.querySelector(".utt-text").textContent;
  openModal("modalEditUtt");
  setTimeout(()=>editText.focus(), 60);
}
["closeEditUtt","cancelEditUtt"].forEach(id => document.getElementById(id)?.addEventListener("click", ()=>closeModal("modalEditUtt")));
modalEditUtt?.addEventListener("click", e=>{ if(e.target===modalEditUtt) closeModal("modalEditUtt"); });

document.getElementById("saveEditUtt")?.addEventListener("click", async () => {
  const id      = editUttId.value;
  const speaker = editSpeaker.value.trim();
  const text    = editText.value.trim();
  const model   = document.getElementById("modelSelect").value;
  if (!text) { toast("Text cannot be empty","error"); return; }
  const btn = document.getElementById("saveEditUtt");
  btn.disabled=true; btn.innerHTML='<span class="spinner"></span>';
  try {
    const r = await fetch(`/api/utterances/${id}`, {
      method:"PUT", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({speaker, text, model})
    });
    const u = await r.json();
    const row = document.querySelector(`.utt-row[data-id="${id}"]`);
    if (row) {
      row.querySelector(".utt-speaker").textContent = u.speaker;
      row.querySelector(".utt-avatar").textContent  = u.speaker[0].toUpperCase();
      document.getElementById(`uttText${id}`).textContent   = u.text;
      document.getElementById(`uttFooter${id}`).innerHTML   = buildEmoBadge(u);
    }
    closeModal("modalEditUtt");
    toast("Updated","success");
  } catch { toast("Update failed","error"); }
  finally { btn.disabled=false; btn.textContent="Save"; }
});

// ── Run Analysis ──────────────────────────────────────────────
const btnAnalyse = document.getElementById("btnAnalyse");
const statPairs  = document.getElementById("statPairs");
const statModel  = document.getElementById("statModel");
const statF1     = document.getElementById("statF1");
const pairsCont  = document.getElementById("tabPairs");

let trendChart = null, distChart = null;

btnAnalyse?.addEventListener("click", async () => {
  if (turnCount < 2) { toast("Add at least 2 turns","error"); return; }
  const model = document.getElementById("modelSelect").value;
  btnAnalyse.disabled=true; btnAnalyse.innerHTML='<span class="spinner"></span> Analysing…';
  try {
    const r = await fetch(`/api/conversations/${CONV_ID}/analyse`, {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({model})
    });
    if (!r.ok) throw new Error((await r.json()).error||"failed");
    const d = await r.json();

    // update emotion badges
    d.utterances.forEach(u => {
      const footer = document.getElementById(`uttFooter${u.utterance_id}`);
      if (footer) footer.innerHTML = buildEmoBadge(u);
    });

    // stats
    statPairs.textContent = d.pairs.length;
    statModel.textContent = model.split(" ")[0];
    // fetch F1 from /api/metrics
    fetch("/api/metrics").then(r=>r.json()).then(m => {
      const f1 = m[model]?.f1;
      if (f1) statF1.textContent = pct(f1)+"%";
    });

    // render pairs
    renderPairs(d);

    // trend chart
    renderTrendChart(d.trend);

    // dist chart
    renderDistChart(d.stats.emotion_distribution);

    toast(`Analysis complete — ${d.pairs.length} pair(s) found`, "success");
    trackEvent("run_analysis", { model: model, pairs_found: d.pairs.length, dominant_emotion: d.stats.dominant_emotion });
  } catch(e) { toast(e.message||"Analysis failed","error"); }
  finally { btnAnalyse.disabled=false; btnAnalyse.textContent="🤖 Run Analysis"; }
});

function renderPairs(data) {
  if (!data.pairs.length) {
    pairsCont.innerHTML = `<div class="empty-pairs"><div class="empty-icon">🔍</div><p>No emotion-cause pairs detected. Try adding more expressive utterances.</p></div>`;
    return;
  }
  pairsCont.innerHTML = data.pairs.map(p => {
    const eu = data.utterances.find(u=>u.turn_index===p.emotion_turn);
    const cu = data.utterances.find(u=>u.turn_index===p.cause_turn);
    const col = p.color||"#94a3b8";
    return `<div class="pair-card">
      <div class="pair-header">
        <span class="pair-emo-badge" style="background:${col}18;color:${col};border-color:${col}35">${p.icon||""} ${cap(p.emotion)}</span>
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
  }).join("");
}

function renderTrendChart(trend) {
  const ctx = document.getElementById("trendChart");
  if (!ctx) return;
  const labels = trend.map(t=>`T${t.turn}`);
  const emoMap = {};
  trend.forEach(t => { emoMap[t.emotion] = (emoMap[t.emotion]||0)+1; });
  const emotions = Object.keys(emoMap);
  const datasets = emotions.map(e => ({
    label: cap(e),
    data: trend.map(t => t.emotion===e ? t.confidence : null),
    borderColor: EMO_COLORS[e]||"#94a3b8",
    backgroundColor: (EMO_COLORS[e]||"#94a3b8")+"33",
    tension: 0.4, spanGaps: true, pointRadius: 5,
  }));
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type:"line",
    data:{ labels, datasets },
    options:{
      responsive:true,plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:"#64748b",font:{size:10}},grid:{color:"#252d47"}},
        y:{min:0,max:1,ticks:{color:"#64748b",font:{size:10},callback:v=>pct({valueOf:()=>v})+"%"},grid:{color:"#252d47"}},
      }
    }
  });
  // legend
  const legend = document.getElementById("trendLegend");
  if (legend) legend.innerHTML = emotions.map(e=>`<span class="legend-item"><span class="legend-dot" style="background:${EMO_COLORS[e]||'#94a3b8'}"></span>${cap(e)}</span>`).join("");
}

function renderDistChart(dist) {
  const ctx = document.getElementById("distChart");
  if (!ctx) return;
  const entries = Object.entries(dist).filter(([e])=>e!=="neutral"||dist[e]>0).sort((a,b)=>b[1]-a[1]);
  if (distChart) distChart.destroy();
  distChart = new Chart(ctx, {
    type:"doughnut",
    data:{
      labels: entries.map(([e])=>cap(e)),
      datasets:[{ data: entries.map(([,n])=>n), backgroundColor: entries.map(([e])=>EMO_COLORS[e]||"#94a3b8"), borderColor:"#111520", borderWidth:2 }]
    },
    options:{ responsive:true, plugins:{ legend:{ labels:{ color:"#94a3b8", font:{size:11} } } } }
  });
  // bar list
  const total = Object.values(dist).reduce((a,b)=>a+b,0);
  const bars  = document.getElementById("distBars");
  if (bars) bars.innerHTML = entries.map(([e,n])=>`<div class="dist-row"><span class="dist-label">${EMO_COLORS[e]?"":""} ${cap(e)}</span><div class="dist-bar"><div class="dist-fill" style="width:${total?Math.round(n/total*100):0}%;background:${EMO_COLORS[e]||'#94a3b8'}"></div></div><span class="dist-count">${n}</span></div>`).join("");
}

// ── Rename ────────────────────────────────────────────────────
const renameInput = document.getElementById("renameInput");
document.getElementById("btnRename")?.addEventListener("click", () => {
  renameInput.value = window.CONV_TITLE;
  openModal("modalRename");
  setTimeout(()=>renameInput.focus(),60);
});
["closeRename","cancelRename"].forEach(id=>document.getElementById(id)?.addEventListener("click",()=>closeModal("modalRename")));
document.getElementById("modalRename")?.addEventListener("click",e=>{if(e.target.id==="modalRename")closeModal("modalRename");});
renameInput?.addEventListener("keydown",e=>{if(e.key==="Enter")document.getElementById("saveRename").click();});

document.getElementById("saveRename")?.addEventListener("click", async () => {
  const title = renameInput.value.trim();
  if (!title) return;
  await fetch(`/api/conversations/${CONV_ID}/rename`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title})});
  document.getElementById("convTitleDisplay").textContent = title;
  window.CONV_TITLE = title; document.title = `${title} — ECPE AI Lab`;
  closeModal("modalRename"); toast("Renamed","success");
});
