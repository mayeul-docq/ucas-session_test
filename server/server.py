#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, math
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.agent_core import MatchingAgent, MAX_QUESTIONS_PER_UNI
from agent.agent_llm import LLMEnhancedAgent

# ---------- Config ----------
API_ALLOW_ORIGINS = ["*"]  # restreins si besoin
TOP_CONFIDENCE_TARGET = 10
CONF_DELTA_EPS = 0.015
CONF_MIN_POINTS = 3

# ---------- Chargement des stores normalisés ----------
def load_store(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        out = []
        for k, v in data.items():
            norm = v.get("normalized") or {}
            if "id" not in norm: norm["id"] = k
            out.append(norm)
        return out
    return data

STUDENTS_PATH = os.environ.get("STUDENTS_PATH", "./normalized/students_normalized.json")
UNIS_PATH = os.environ.get("UNIS_PATH", "./normalized/universities_normalized.json")
STUDENTS = load_store(STUDENTS_PATH)
UNIS = load_store(UNIS_PATH)

if not UNIS:
    raise RuntimeError("No universities found in ./normalized/universities_normalized.json")

# par défaut: premier étudiant
DEFAULT_STUDENT = STUDENTS[0] if STUDENTS else {"id":"demo","preferences":{}}

# ---------- Sessions en mémoire ----------
class Session:
    def __init__(self, student: Dict[str, Any], api_key: Optional[str]):
        self.api_key = api_key
        self.agent = LLMEnhancedAgent(student, UNIS, api_key) if api_key else MatchingAgent(student, UNIS)
        self.comments: Dict[str, List[str]] = {uid: [] for uid in self.agent.uni_ids}
        self.questions_count: Dict[str, int] = {uid: 0 for uid in self.agent.uni_ids}
        self.asked_questions: Dict[str, List[Dict[str,str]]] = {uid: [] for uid in self.agent.uni_ids}
        self.scores_history: Dict[str, List[float]] = {uid: [] for uid in self.agent.uni_ids}
        self.confident_unis: set[str] = set()

    def current_state(self) -> Dict[str, Any]:
        st = self.agent.step()
        # enregistre historique des scores pour la confiance
        for uid, sc in st["scores"].items():
            self.scores_history[uid].append(sc["hybrid"])
            self.scores_history[uid] = self.scores_history[uid][-5:]
        # calcule confiance
        self._update_confidence()
        st["confident_unis"] = list(self.confident_unis)
        st["should_stop"] = self.should_stop()
        return st

    def _update_confidence(self):
        self.confident_unis = set()
        for uid, hist in self.scores_history.items():
            if len(hist) >= CONF_MIN_POINTS:
                deltas = [abs(hist[i]-hist[i-1]) for i in range(1, len(hist))]
                if len(deltas) >= CONF_MIN_POINTS-1 and max(deltas[-(CONF_MIN_POINTS-1):]) <= CONF_DELTA_EPS:
                    self.confident_unis.add(uid)

    def should_stop(self) -> bool:
        total_unis = len(self.agent.uni_ids)
        target = min(TOP_CONFIDENCE_TARGET, total_unis)
        # stop si toutes les questions ont été posées pour toutes les unis du triplet
        all_questions_done = all(self.questions_count[uid] >= MAX_QUESTIONS_PER_UNI for uid in self.agent.triplet)
        # ou confiance atteinte
        conf_ok = len(self.confident_unis) >= (target if total_unis >= 10 else total_unis)
        return all_questions_done or conf_ok

# sessions par student_id
SESSIONS: Dict[str, Session] = {}

# ---------- FastAPI ----------
app = FastAPI(title="UNIVIA Agent API")
app.add_middleware(CORSMiddleware, allow_origins=API_ALLOW_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------- Schemas ----------
class InitPayload(BaseModel):
    student_id: Optional[str] = None
    openai_api_key: Optional[str] = None  # fourni par l'élève

class CommentPayload(BaseModel):
    student_id: str
    uni_id: str
    text: str

class AnswerPayload(BaseModel):
    student_id: str
    uni_id: str
    slot: str
    value: str

class PairwisePayload(BaseModel):
    student_id: str
    better_id: str
    worse_id: str

# ---------- Helpers ----------
def get_student_by_id(sid: str) -> Dict[str, Any]:
    for s in STUDENTS:
        if s.get("id") == sid: return s
    return DEFAULT_STUDENT

def ensure_session(sid: str) -> Session:
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "Session not found. Call /api/init first.")
    return sess

# ---------- Endpoints ----------
@app.post("/api/init")
def api_init(body: InitPayload):
    sid = body.student_id or (DEFAULT_STUDENT.get("id") if DEFAULT_STUDENT else "demo")
    student = get_student_by_id(sid)
    sess = Session(student, body.openai_api_key)
    SESSIONS[sid] = sess
    return {"ok": True, "student_id": sid, "state": sess.current_state()}

@app.get("/api/state")
def api_state(student_id: str):
    sess = ensure_session(student_id)
    return {"ok": True, "state": sess.current_state()}

@app.post("/api/comment")
def api_comment(body: CommentPayload):
    sess = ensure_session(body.student_id)
    if body.uni_id not in sess.comments: sess.comments[body.uni_id] = []
    sess.comments[body.uni_id].append(body.text)

    # les questions apparaissent APRÈS le 1er commentaire
    questions = []
    if len(sess.comments[body.uni_id]) == 1:
        # propose questions (LLM si dispo, sinon core)
        if hasattr(sess.agent, "suggest_questions_for_triplet"):
            llm_qs = sess.agent.suggest_questions_for_triplet(max_q=3)
            # map slot->texte lisible
            slot2txt = {
                "budget_range": "Quel budget annuel total (frais + logement + vie) visez-vous (EUR) ?",
                "need_portfolio": "Aurez-vous un portfolio prêt d’ici la deadline UCAS ?",
                "ielts_plan": "Avez-vous un plan pour l’IELTS (date visée, score cible) ?",
                "campus_setting": "Préférez-vous un campus urbain, suburbain ou rural ?",
                "pmr_needs": "Avez-vous des besoins d’accessibilité (PMR) à prendre en compte ?"
            }
            for q in llm_qs:
                if sess.questions_count[body.uni_id] < MAX_QUESTIONS_PER_UNI:
                    questions.append({"slot": q["slot"], "text": slot2txt.get(q["slot"], q["slot"])})
                    sess.questions_count[body.uni_id] += 1
        else:
            # fallback core
            for q in sess.agent.next_questions(max_q=3):
                if sess.questions_count[body.uni_id] < MAX_QUESTIONS_PER_UNI:
                    questions.append(q)
                    sess.questions_count[body.uni_id] += 1

        sess.asked_questions[body.uni_id].extend(questions)

    return {"ok": True, "questions": questions, "state": sess.current_state()}

@app.post("/api/answer")
def api_answer(body: AnswerPayload):
    sess = ensure_session(body.student_id)
    # applique la réponse (met à jour le profil élève)
    sess.agent.apply_answers({body.slot: body.value})

    # pose une prochaine question si quota non atteint (≤5)
    questions = []
    if sess.questions_count[body.uni_id] < MAX_QUESTIONS_PER_UNI:
        nxt = sess.agent.next_questions(max_q=1)
        if nxt:
            questions.append(nxt[0])
            sess.questions_count[body.uni_id] += 1
            sess.asked_questions[body.uni_id].extend(questions)

    # Remplacement de colonnes si besoin (simple : on remplace la moins bien notée du triplet par une non vue)
    # seulement si pas en stop
    if not sess.should_stop():
        trip = list(sess.agent.triplet)
        # calc des scores
        scored = [(uid, sess.agent.hybrid_score(uid)) for uid in trip]
        worst = min(scored, key=lambda x: x[1])[0]
        # pick best candidate non présent
        others = [u for u in sess.agent.uni_ids if u not in trip]
        if others:
            others.sort(key=lambda x: sess.agent.hybrid_score(x), reverse=True)
            replacement = others[0]
            idx = trip.index(worst)
            trip[idx] = replacement
            sess.agent.triplet = trip

    return {"ok": True, "questions": questions, "state": sess.current_state()}

@app.post("/api/pairwise")
def api_pairwise(body: PairwisePayload):
    sess = ensure_session(body.student_id)
    sess.agent.feedback_pairwise(body.better_id, body.worse_id)
    return {"ok": True, "state": sess.current_state()}

@app.get("/api/ranking")
def api_ranking(student_id: str):
    sess = ensure_session(student_id)
    # classement final par score hybride
    ranked = sorted(sess.agent.uni_ids, key=lambda x: sess.agent.hybrid_score(x), reverse=True)
    table = [{"uni_id": u, "score": round(sess.agent.hybrid_score(u), 3)} for u in ranked]
    stop = sess.should_stop()
    return {"ok": True, "stop": stop, "ranking": table}
