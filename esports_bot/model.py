"""Модель прогнозов: рейтинг Эло + текущая форма + личные встречи."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

BASE_RATING = 1500.0
BASE_K = 32.0
# Матчи топ-турниров влияют на рейтинг сильнее
TIER_WEIGHT = {"s": 1.25, "a": 1.1, "b": 1.0, "c": 0.85, "d": 0.7}
FORM_WINDOW = 10   # сколько последних матчей считать «формой»
H2H_WINDOW = 5     # сколько последних личных встреч учитывать


def _ts(m: dict) -> str:
    return m.get("end_at") or m.get("begin_at") or m.get("scheduled_at") or ""


def _teams(m: dict) -> list[dict]:
    return [o["opponent"] for o in m.get("opponents") or [] if o.get("opponent")]


def _score(m: dict, team_id: int) -> int:
    for r in m.get("results") or []:
        if r.get("team_id") == team_id:
            return r.get("score") or 0
    return 0


def elo_expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


@dataclass
class TeamStats:
    id: int
    name: str
    acronym: str = ""
    rating: float = BASE_RATING
    played: int = 0
    results: list[tuple[str, int, bool]] = field(default_factory=list)  # (дата, id соперника, победа)

    @property
    def recent(self) -> list[tuple[str, int, bool]]:
        return self.results[-FORM_WINDOW:]

    def form(self) -> float | None:
        """Доля побед в последних матчах с весом свежести (новые важнее)."""
        rec = self.recent
        if not rec:
            return None
        weights = [0.85 ** (len(rec) - 1 - i) for i in range(len(rec))]
        return sum(w for w, (_, _, win) in zip(weights, rec) if win) / sum(weights)

    def streak(self) -> str:
        return "".join("W" if win else "L" for _, _, win in self.recent[-5:])


@dataclass
class Prediction:
    a: TeamStats
    b: TeamStats
    p_a: float
    p_elo: float
    h2h: tuple[int, int]
    confidence: str
    factors: list[str]


class EloModel:
    def __init__(self) -> None:
        self.teams: dict[int, TeamStats] = {}
        self.updated_at: datetime | None = None
        self.matches_used = 0

    def _team(self, t: dict) -> TeamStats:
        ts = self.teams.get(t["id"])
        if ts is None:
            ts = self.teams[t["id"]] = TeamStats(t["id"], t.get("name") or t.get("acronym") or "?",
                                                   t.get("acronym") or "")
        return ts

    def fit(self, past_matches: list[dict]) -> None:
        self.teams.clear()
        self.matches_used = 0
        for m in sorted(past_matches, key=_ts):
            teams = _teams(m)
            winner = m.get("winner_id")
            if len(teams) != 2 or winner is None or m.get("forfeit"):
                continue
            a, b = self._team(teams[0]), self._team(teams[1])
            if winner not in (a.id, b.id):
                continue
            a_won = winner == a.id
            exp_a = elo_expect(a.rating, b.rating)

            tier = ((m.get("tournament") or {}).get("tier") or "b").lower()
            k = BASE_K * TIER_WEIGHT.get(tier, 1.0)
            # Разгромы (2:0) двигают рейтинг сильнее, чем 2:1
            sa, sb = _score(m, a.id), _score(m, b.id)
            if sa + sb > 1:
                k *= 1.0 + 0.5 * abs(sa - sb) / (sa + sb)
            # Новые команды «калибруются» быстрее
            k *= 1.5 if min(a.played, b.played) < 5 else 1.0

            delta = k * ((1.0 if a_won else 0.0) - exp_a)
            a.rating += delta
            b.rating -= delta
            a.played += 1
            b.played += 1
            when = _ts(m)
            a.results.append((when, b.id, a_won))
            b.results.append((when, a.id, not a_won))
            self.matches_used += 1
        self.updated_at = datetime.now()

    def get_or_new(self, t: dict) -> TeamStats:
        return self.teams.get(t["id"]) or TeamStats(t["id"], t.get("name") or "?")

    def h2h(self, a: TeamStats, b: TeamStats) -> tuple[int, int]:
        games = [win for _, opp, win in a.results if opp == b.id][-H2H_WINDOW:]
        return sum(games), len(games) - sum(games)

    def predict(self, team_a: dict, team_b: dict, best_of: int | None = None) -> Prediction:
        a, b = self.get_or_new(team_a), self.get_or_new(team_b)
        p_elo = elo_expect(a.rating, b.rating)
        logit = math.log(p_elo / (1 - p_elo))
        factors: list[str] = []

        fa, fb = a.form(), b.form()
        if fa is not None and fb is not None:
            logit += 0.4 * (fa - fb)
            if abs(fa - fb) >= 0.25:
                better = a if fa > fb else b
                factors.append(f"{better.name} в заметно лучшей форме")

        wa, wb = self.h2h(a, b)
        if wa + wb:
            # сглаживание: 2-3 встречи не должны перевешивать рейтинг
            shrunk = (wa + 1) / (wa + wb + 2)
            logit += 0.4 * (shrunk - 0.5)
            if wa != wb:
                factors.append(f"Личные встречи: {a.name} {wa}–{wb} {b.name}")

        # В Bo1 больше случайности — прижимаем прогноз к 50%
        if best_of == 1:
            logit *= 0.8
            factors.append("Bo1 — высокая доля случайности")
        elif best_of and best_of >= 5:
            logit *= 1.1

        p_a = 1 / (1 + math.exp(-logit))

        rd = a.rating - b.rating
        if abs(rd) >= 60:
            fav = a if rd > 0 else b
            factors.insert(0, f"{fav.name} выше по рейтингу на {abs(rd):.0f}")

        min_played = min(a.played, b.played)
        if min_played < 5:
            confidence = "низкая (мало данных о команде)"
        elif min_played < 15 or abs(p_a - 0.5) < 0.08:
            confidence = "средняя"
        else:
            confidence = "высокая"

        return Prediction(a, b, p_a, p_elo, (wa, wb), confidence, factors)

    def find_team(self, query: str) -> list[TeamStats]:
        q = query.lower().strip()
        exact = [t for t in self.teams.values() if t.name.lower() == q]
        if exact:
            return exact
        return sorted((t for t in self.teams.values() if q in t.name.lower()),
                      key=lambda t: -t.played)[:5]

    def top(self, n: int = 15, min_played: int = 8) -> list[TeamStats]:
        return sorted((t for t in self.teams.values() if t.played >= min_played),
                      key=lambda t: -t.rating)[:n]
