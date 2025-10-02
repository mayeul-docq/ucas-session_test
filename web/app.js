// Config API (local FastAPI)
const API_BASE = (localStorage.getItem("API_BASE") || "http://127.0.0.1:8000").replace(/\/+$/,"");
document.getElementById("apiBase").textContent = API_BASE;

const elTriplet = document.getElementById("triplet");
const elRanking = document.getElementById("ranking");
const elRankingList = document.getElementById("rankingList");

let SESSION = {
  studentId: null,
  apiKey: null,
  state: null,       // retour complet backend
  answers: {},       // slot -> value (global)
  comments: {},      // uni_id -> first comment text
};

function badgeScore(v){
  if (v >= 0.8) return "score ok";
  if (v >= 0.6) return "score warn";
  return "score bad";
}

async function api(path, method="GET", body=null){
  const url = `${API_BASE}${path}`;
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const res = await fetch(url, opt);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function renderTriplet(state){
  elTriplet.innerHTML = "";
  const trip = state.triplet || [];
  trip.forEach(uid => {
    const sc = state.scores[uid];
    const score = sc ? sc.hybrid : 0.0;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="meta">
        <span class="badge">ID: ${uid}</span>
        <span class="badge">Soft: ${sc ? sc.soft_fit.toFixed(2) : "-"}</span>
        <span class="badge">Pref: ${sc ? Math.round(sc.pref) : "-"}</span>
      </div>
      <h3>Université: ${uid}</h3>
      <div class="${badgeScore(score)}">Score: ${score.toFixed(3)}</div>
      <textarea class="comment" placeholder="Votre commentaire sur cette université..." data-uid="${uid}">${SESSION.comments[uid] || ""}</textarea>
      <div class="actions-row">
        <button class="btn primary" data-action="sendComment" data-uid="${uid}">Envoyer le commentaire</button>
        <button class="btn ghost" data-action="prefer0" data-uid="${uid}">Préférer cette colonne à la 1</button>
      </div>
      <div class="q-list" id="q-${uid}"></div>
    `;
    elTriplet.appendChild(card);
  });
}

function wireButtons(){
  elTriplet.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const uid = btn.dataset.uid;
    const action = btn.dataset.action;

    if (action === "sendComment"){
      const ta = elTriplet.querySelector(`textarea[data-uid="${uid}"]`);
      const text = (ta.value || "").trim();
      if (!text){ alert("Merci d'écrire un commentaire."); return; }
      SESSION.comments[uid] = text;
      const res = await api(`/api/comment`, "POST", { student_id: SESSION.studentId, uni_id: uid, text });
      SESSION.state = res.state;
      renderTriplet(SESSION.state);
      // afficher questions
      const qs = res.questions || [];
      const list = document.getElementById(`q-${uid}`);
      list.innerHTML = "";
      qs.forEach(q => {
        const row = document.createElement("div");
        row.className = "q-item";
        row.innerHTML = `
          <span>Q:</span>
          <input placeholder="${q.text}" data-slot="${q.slot}" />
          <button class="btn" data-action="answer" data-uid="${uid}" data-slot="${q.slot}">OK</button>
        `;
        list.appendChild(row);
      });
    }

    if (action === "answer"){
      const slot = btn.dataset.slot;
      const input = btn.previousElementSibling;
      const value = (input.value || "").trim();
      if (!value){ alert("Merci de répondre."); return; }
      SESSION.answers[slot] = value;
      const res = await api(`/api/answer`, "POST", { student_id: SESSION.studentId, uni_id: uid, slot, value });
      SESSION.state = res.state;
      renderTriplet(SESSION.state);
      // On affiche 0 ou 1 nouvelle question possible (géré côté serveur)
      const qs = res.questions || [];
      const list = document.getElementById(`q-${uid}`);
      qs.forEach(q => {
        const row = document.createElement("div");
        row.className = "q-item";
        row.innerHTML = `
          <span>Q:</span>
          <input placeholder="${q.text}" data-slot="${q.slot}" />
          <button class="btn" data-action="answer" data-uid="${uid}" data-slot="${q.slot}">OK</button>
        `;
        list.appendChild(row);
      });
    }

    if (action === "prefer0"){
      // préfère cette colonne à la 1ère (si différente)
      const trip = (SESSION.state.triplet || []);
      if (trip.length < 2) return;
      const a = uid, b = trip[0] === uid ? trip[1] : trip[0];
      const res = await api(`/api/pairwise`, "POST", { student_id: SESSION.studentId, better_id: a, worse_id: b });
      SESSION.state = res.state;
      renderTriplet(SESSION.state);
    }
  });

  document.getElementById("refreshState").addEventListener("click", async () => {
    const res = await api(`/api/state?student_id=${encodeURIComponent(SESSION.studentId)}`);
    SESSION.state = res.state;
    renderTriplet(SESSION.state);
  });

  document.getElementById("showRanking").addEventListener("click", async () => {
    const res = await api(`/api/ranking?student_id=${encodeURIComponent(SESSION.studentId)}`);
    elRanking.classList.remove("hidden");
    elRankingList.innerHTML = "";
    res.ranking.forEach((r, i) => {
      const li = document.createElement("li");
      li.textContent = `${i+1}. ${r.uni_id} — ${r.score.toFixed(3)}`;
      elRankingList.appendChild(li);
    });
    if (res.stop) {
      const li = document.createElement("li");
      li.innerHTML = `<strong>⚑ Arrêt atteint (confiance des scores).</strong>`;
      elRankingList.appendChild(li);
    }
  });
}

async function start(){
  const studentId = document.getElementById("studentId").value.trim() || null;
  const apiKey = document.getElementById("apiKey").value.trim() || null;
  const res = await api(`/api/init`, "POST", { student_id: studentId, openai_api_key: apiKey });
  SESSION.studentId = res.student_id;
  SESSION.apiKey = apiKey;
  SESSION.state = res.state;
  renderTriplet(SESSION.state);
}

document.getElementById("startBtn").addEventListener("click", start);
wireButtons();
