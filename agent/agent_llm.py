#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, requests
from typing import Any, Dict, List, Optional
from agent.agent_core import MatchingAgent

SCORING_SYSTEM = """
Tu es un assistant de scoring pour un élève et une université.
Rends un JSON STRICT:
{
  "soft_fit_adjusted": number,
  "explanations": { "pro": [string], "cons": [string] },
  "missing_slots": [ "budget_range" | "need_portfolio" | "ielts_plan" | "campus_setting" | "pmr_needs" ],
  "decision_hint": "go" | "maybe" | "no-go"
}
"""

class LLMClient:
    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 60) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "response_format":{"type":"json_object"}, "temperature": temperature, "messages": messages}
        r = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)

class LLMEnhancedAgent(MatchingAgent):
    def __init__(self, student: Dict[str, Any], universities: List[Dict[str, Any]], api_key: Optional[str]):
        super().__init__(student, universities)
        self.llm = LLMClient(api_key) if api_key else None

    def _llm_adjust(self, uid: str, deterministic_score: float) -> Dict[str, Any]:
        if not self.llm:
            return {"soft_fit_adjusted": deterministic_score, "explanations":{"pro":[],"cons":[]}, "missing_slots": [], "decision_hint": "maybe"}
        sys_msg = {"role":"system","content":SCORING_SYSTEM}
        user_msg = {"role":"user","content":json.dumps({"student": self.student, "university": self.unis[uid], "deterministic_soft_fit": deterministic_score}, ensure_ascii=False)}
        try:
            return self.llm.chat_json([sys_msg, user_msg])
        except Exception:
            return {"soft_fit_adjusted": deterministic_score, "explanations":{"pro":[],"cons":[]}, "missing_slots": [], "decision_hint": "maybe"}

    def suggest_questions_for_triplet(self, max_q: int = 3) -> List[Dict[str, str]]:
        # agrège les slots manquants suggérés par LLM sur le triplet
        out = []
        for uid in self.triplet:
            sf, _ = self.score_uni(uid)
            llm = self._llm_adjust(uid, sf)
            for slot in llm.get("missing_slots", []):
                if slot not in [q["slot"] for q in out]:
                    out.append({"slot": slot, "text": slot})  # le serveur mappe slot->texte final
        return out[:max_q]
