# 🤖 `agent/agent_core.py`

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, math, random, statistics
from typing import Any, Dict, List, Optional, Tuple, Set

FIT_THRESHOLD_TOP5 = 0.72
ELO_K = 24.0
BTL_LR = 0.08
INITIAL_PREF = 1000.0
ALPHA_START = 0.65

SLOTS = ["budget_range","need_portfolio","ielts_plan","campus_setting","pmr_needs"]
MAX_QUESTIONS_PER_UNI = 5  # borne localement ; le serveur applique aussi

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(p)
        if cur is None: return default
    return cur

def money_amount(d: Optional[Dict[str, Any]]) -> Optional[float]:
    if not d: return None
    return d.get("amount")

def soft_fit(student: Dict[str, Any], uni: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    w = {"accreditation":0.24,"portfolio":0.16,"campus_setting":0.12,"budget_gap":0.18,"student_staff_ratio":0.10,"english_ready":0.10,"accessibility_pmr":0.10}
    breakdown = {k: 0.0 for k in w}

    acc = set(safe_get(uni, ["offer","accreditations"], []) or [])
    breakdown["accreditation"] = 1.0 if ("RIBA" in acc or "ARB" in acc) else 0.0

    requires_portfolio = bool(safe_get(uni, ["admissions","requires_portfolio"], False))
    arts_grade = safe_get(student, ["academics","grades","arts_plastiques"], None)
    portfolio_ready_hint = arts_grade is not None and arts_grade >= 14
    breakdown["portfolio"] = 1.0 if (not requires_portfolio or portfolio_ready_hint) else 0.4

    pref_setting = safe_get(student, ["preferences","campus_setting"], None)
    campus_setting = safe_get(uni, ["campus","setting"], None)
    if pref_setting and campus_setting:
        breakdown["campus_setting"] = 1.0 if pref_setting == campus_setting else (0.5 if (pref_setting in {"urban","suburban"} and campus_setting in {"urban","suburban"}) else 0.0)
    else:
        breakdown["campus_setting"] = 0.5

    budget_eur = safe_get(student, ["budget","annual_total","amount"], None)
    tuition = money_amount(safe_get(uni, ["fees","tuition"], None))
    living_gbp = 12000.0 + (3000.0 if campus_setting == "urban" else 0.0)
    if tuition is None or budget_eur is None:
        budget_score = 0.6
    else:
        total_eur = (tuition + living_gbp) * 1.15
        gap = budget_eur - total_eur
        if gap >= 5000: budget_score = 1.0
        elif gap >= 0: budget_score = 0.8
        elif gap > -5000: budget_score = 0.5
        else: budget_score = 0.2
    breakdown["budget_gap"] = budget_score

    ssr = safe_get(uni, ["offer","student_staff_ratio"], None)
    if ssr is None:
        ssr_score = 0.6
    else:
        if ssr <= 12: ssr_score = 1.0
        elif ssr <= 16: ssr_score = 0.8
        elif ssr <= 20: ssr_score = 0.6
        else: ssr_score = 0.4
    breakdown["student_staff_ratio"] = ssr_score

    ielts_needed = safe_get(uni, ["admissions","english_min","ielts_overall"], None)
    ielts_score = safe_get(student, ["academics","english","score"], None)
    if ielts_needed is None: eng_score = 0.7
    else:
        if ielts_score is None: eng_score = 0.6
        else:
            diff = ielts_score - ielts_needed
            if diff >= 0.5: eng_score = 1.0
            elif diff >= 0.0: eng_score = 0.8
            elif diff >= -0.5: eng_score = 0.6
            else: eng_score = 0.3
    breakdown["english_ready"] = eng_score

    pmr_need = bool(safe_get(student, ["constraints","pmr"], False))
    pmr_ok = bool(safe_get(uni, ["campus","pmr_ok"], False))
    breakdown["accessibility_pmr"] = 1.0 if (not pmr_need or pmr_ok) else 0.3

    score = sum(w[k] * breakdown[k] for k in w)
    return score, breakdown

class PreferenceModel:
    def __init__(self, uni_ids: List[str], use_btl: bool = True):
        self.elo: Dict[str, float] = {u: INITIAL_PREF for u in uni_ids}
        self.btl: Dict[str, float] = {u: 0.0 for u in uni_ids}
        self.use_btl = use_btl

    def ensure(self, uid: str):
        if uid not in self.elo: self.elo[uid] = INITIAL_PREF
        if uid not in self.btl: self.btl[uid] = 0.0

    def prob_btl(self, i: str, j: str) -> float:
        si, sj = self.btl[i], self.btl[j]
        return 1.0 / (1.0 + math.exp(-(si - sj)))

    def update_pair(self, winner: str, loser: str):
        self.ensure(winner); self.ensure(loser)
        rw, rl = self.elo[winner], self.elo[loser]
        Ew = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        self.elo[winner] = rw +  ELO_K * (1.0 - Ew)
        self.elo[loser]  = rl +  ELO_K * (0.0 - (1.0 - Ew))
        if self.use_btl:
            pi = self.prob_btl(winner, loser)
            self.btl[winner] += BTL_LR * (1.0 - pi)
            self.btl[loser]  -= BTL_LR * (1.0 - pi)

class MatchingAgent:
    """
    Core agent: état, soft-fit, ELO/BTL, sélection triplet, slots/questions.
    Le serveur gère la limite de 5 questions par université + l'arrêt par confiance.
    """
    def __init__(self, student: Dict[str, Any], universities: List[Dict[str, Any]], rnd_seed: int = 42):
        self.student = student
        self.unis = {u["id"]: u for u in universities}
        self.uni_ids = list(self.unis.keys())
        random.Random(rnd_seed).shuffle(self.uni_ids)

        self.pref = PreferenceModel(self.uni_ids, use_btl=True)
        self.alpha = ALPHA_START

        self.seen: Set[str] = set()
        self.shortlist: List[str] = []
        self.exclusions: List[Dict[str, str]] = []
        self.asked_slots: Set[str] = set()

        self.per_uni_questions_count: Dict[str, int] = {uid: 0 for uid in self.uni_ids}
        self.soft_cache: Dict[str, float] = {}
        self.hybrid_cache: Dict[str, float] = {}

        self.triplet: List[str] = self._select_initial_triplet()

    # ---- scoring ----
    def score_uni(self, uid: str) -> Tuple[float, Dict[str, float]]:
        sc, br = soft_fit(self.student, self.unis[uid])
        self.soft_cache[uid] = sc
        return sc, br

    def hybrid_score(self, uid: str) -> float:
        sf, _ = self.score_uni(uid)
        elo = self.pref.elo.get(uid, INITIAL_PREF)
        elo_vals = list(self.pref.elo.values())
        if len(elo_vals) >= 2:
            mn, mx = min(elo_vals), max(elo_vals)
            elo01 = (elo - mn) / (mx - mn + 1e-6)
        else:
            elo01 = 0.5
        btl_si = self.pref.btl.get(uid, 0.0)
        btl01 = 1.0 / (1.0 + math.exp(-btl_si))
        pref01 = 0.5 * elo01 + 0.5 * btl01
        h = self.alpha * sf + (1.0 - self.alpha) * pref01
        self.hybrid_cache[uid] = h
        return h

    # ---- selection ----
    def _diversify_pick(self, cand_ids: List[str], k: int) -> List[str]:
        selected: List[str] = []
        seen_keys: Set[tuple] = set()
        for uid in cand_ids:
            uni = self.unis[uid]
            setting = (uni.get("campus") or {}).get("setting") or "na"
            tuition = money_amount((uni.get("fees") or {}).get("tuition"))
            bucket = "high" if (tuition or 0) >= 28000 else "mid" if (tuition or 0) >= 22000 else "low"
            key = (setting, bucket)
            if key in seen_keys and len(selected) < k - 1:
                continue
            selected.append(uid)
            seen_keys.add(key)
            if len(selected) == k: break
        i = 0
        while len(selected) < k and i < len(cand_ids):
            if cand_ids[i] not in selected: selected.append(cand_ids[i])
            i += 1
        return selected[:k]

    def _select_initial_triplet(self) -> List[str]:
        scored = sorted(self.uni_ids, key=lambda x: self.score_uni(x)[0], reverse=True)
        return self._diversify_pick(scored, 3)

    def _rank_all_hybrid(self) -> List[str]:
        ids = list(self.uni_ids)
        ids.sort(key=lambda x: self.hybrid_score(x), reverse=True)
        return ids

    # ---- questions / slots (génériques) ----
    def next_questions(self, max_q: int = 3) -> List[Dict[str, str]]:
        qs: List[Dict[str, str]] = []
        def need(slot: str) -> bool:
            return slot not in self.asked_slots
        if need("budget_range"):
            qs.append({"slot":"budget_range","text":"Quel budget annuel total (frais + logement + vie) visez-vous (EUR) ?"})
        if need("need_portfolio"):
            qs.append({"slot":"need_portfolio","text":"Aurez-vous un portfolio prêt d’ici la deadline UCAS ?"})
        if need("ielts_plan"):
            qs.append({"slot":"ielts_plan","text":"Avez-vous un plan pour l’IELTS (date visée, score cible) ?"})
        if need("campus_setting"):
            qs.append({"slot":"campus_setting","text":"Préférez-vous un campus urbain, suburbain ou rural ?"})
        if need("pmr_needs"):
            qs.append({"slot":"pmr_needs","text":"Avez-vous des besoins d’accessibilité (PMR) à prendre en compte ?"})
        return qs[:max_q]

    def apply_answers(self, answers: Dict[str, Any]):
        for slot, value in answers.items():
            self.asked_slots.add(slot)
            if slot == "budget_range":
                try:
                    amt = float(str(value).replace(",", "."))
                    self.student.setdefault("budget", {})["annual_total"] = {"amount": amt, "currency": "EUR"}
                except Exception:
                    pass
                self.soft_cache.clear()
            elif slot == "need_portfolio":
                self.soft_cache.clear()
            elif slot == "campus_setting":
                val = str(value).strip().lower()
                if val in {"urban","suburban","rural"}:
                    self.student.setdefault("preferences", {})["campus_setting"] = val
                    self.soft_cache.clear()
            elif slot == "pmr_needs":
                flag = str(value).strip().lower() in {"yes","oui","true","vrai"} or value is True
                self.student.setdefault("constraints", {})["pmr"] = bool(flag)
                self.soft_cache.clear()

    # ---- feedback ----
    def feedback_pairwise(self, better_id: str, worse_id: str):
        self.pref.update_pair(winner=better_id, loser=worse_id)
        self.alpha = max(0.35, self.alpha - 0.03)
        self.hybrid_cache.clear()

    # ---- one step ----
    def step(self) -> Dict[str, Any]:
        for uid in self.triplet: self.seen.add(uid)
        # Remplacement géré par le serveur après calcul des questions/stop selon tes règles.
        state = {
            "triplet": self.triplet,
            "scores": {u: {"soft_fit": self.score_uni(u)[0], "pref": self.pref.elo.get(u, INITIAL_PREF), "hybrid": self.hybrid_score(u)} for u in self.triplet},
            "shortlist": self.shortlist,
            "exclusions": self.exclusions,
            "seen_count": len(self.seen),
        }
        return state
