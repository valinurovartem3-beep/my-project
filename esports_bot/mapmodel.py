"""Прогноз по картам CS2: рейтинг на уровне карт + поправка команды на каждую карту.

Сила команды на карте = общий рейтинг (по всем сыгранным картам) + личная
поправка на эту карту. Поправка растёт, только если команда на этой карте
стабильно играет лучше или хуже, чем ожидается от её общего уровня.
Так 1–2 случайных результата не делают карту «коронной».
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

BASE = 1500.0
K_TEAM = 18.0
K_MAP = 14.0
TIER_WEIGHT = {"S": 1.2, "A": 1.0, "B": 0.85}
STOPWORDS = {"team", "the", "esports", "esport", "gaming", "clan", "club", "gg"}


def norm(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.lower().replace(".", ""))
    core = [w for w in words if w not in STOPWORDS] or words
    return "".join(core)


def expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


@dataclass
class MapRecord:
    played: int = 0
    wins: int = 0
    rounds_won: int = 0
    rounds_total: int = 0
    t_won: int = 0
    t_total: int = 0
    ct_won: int = 0
    ct_total: int = 0
    offset: float = 0.0
    last: str = ""


@dataclass
class MapTeam:
    slug: str
    rating: float = BASE
    played: int = 0
    maps: dict[str, MapRecord] = field(default_factory=lambda: defaultdict(MapRecord))


@dataclass
class MapPrediction:
    pool: list[str]
    per_map: dict[str, float]          # P(команда A выигрывает карту)
    veto: list[tuple[str, str, str]]   # (кто, действие, карта)
    played_maps: list[str]             # ожидаемый порядок карт
    series_p: float
    scores: dict[str, float]           # «2:0» -> вероятность


class MapModel:
    def __init__(self) -> None:
        self.teams: dict[str, MapTeam] = {}
        self.pool: list[str] = []
        self.maps_used = 0

    def fit(self, maps: list[dict]) -> None:
        self.teams.clear()
        for m in sorted(maps, key=lambda x: x["date"] or ""):
            a = self.teams.setdefault(m["team1"], MapTeam(m["team1"]))
            b = self.teams.setdefault(m["team2"], MapTeam(m["team2"]))
            ra, rb = a.maps[m["map"]], b.maps[m["map"]]
            exp_a = expect(a.rating + ra.offset, b.rating + rb.offset)
            won_a = m["score1"] > m["score2"]
            err = (1.0 if won_a else 0.0) - exp_a
            w = TIER_WEIGHT.get(m.get("tier") or "", 0.9)
            # разница в раундах: 13:3 весомее, чем 13:11
            margin = abs(m["score1"] - m["score2"]) / max(m["score1"] + m["score2"], 1)
            k_mult = w * (0.75 + margin) * (1.5 if min(a.played, b.played) < 10 else 1.0)
            a.rating += K_TEAM * k_mult * err
            b.rating -= K_TEAM * k_mult * err
            ra.offset += K_MAP * w * err
            rb.offset -= K_MAP * w * err
            for team, rec, won, own, opp, sides in (
                    (a, ra, won_a, m["score1"], m["score2"], m["t1sides"]),
                    (b, rb, not won_a, m["score2"], m["score1"], m["t2sides"])):
                team.played += 1
                rec.played += 1
                rec.wins += won
                rec.rounds_won += own
                rec.rounds_total += own + opp
                if sides[1] and sides[3]:
                    rec.t_won += sides[0]; rec.t_total += sides[1]
                    rec.ct_won += sides[2]; rec.ct_total += sides[3]
                rec.last = m["date"] or rec.last
        self.maps_used = len(maps)
        # актуальный маппул — 7 самых частых карт за последние 90 дней
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        counts = Counter(m["map"] for m in maps if (m["date"] or "") >= cutoff) or \
            Counter(m["map"] for m in maps)
        self.pool = [name for name, _ in counts.most_common(7)]

    # ---------- поиск команд ----------

    def resolve(self, *names: str) -> MapTeam | None:
        """Находит команду Liquipedia по названию/аббревиатуре из PandaScore или от пользователя."""
        keys = [norm(n) for n in names if n]
        index = {norm(slug): t for slug, t in self.teams.items()}
        for k in keys:
            if k in index:
                return index[k]
        for k in keys:
            if len(k) >= 4:
                hits = [t for nk, t in index.items() if k in nk or (len(nk) >= 4 and nk in k)]
                if hits:
                    return max(hits, key=lambda t: t.played)
        return None

    # ---------- прогноз ----------

    def p_map(self, a: MapTeam, b: MapTeam, name: str) -> float:
        oa = a.maps[name].offset if name in a.maps else 0.0
        ob = b.maps[name].offset if name in b.maps else 0.0
        return expect(a.rating + oa, b.rating + ob)

    def _comfort(self, t: MapTeam, p_win: float, name: str) -> float:
        """Насколько команде хочется играть карту: шанс победы + опыт на ней."""
        played = t.maps[name].played if name in t.maps else 0
        p = min(max(p_win, 0.02), 0.98)
        return math.log(p / (1 - p)) + 0.35 * math.log1p(played) - (0.8 if played == 0 else 0.0)

    def simulate_veto(self, a: MapTeam, b: MapTeam, best_of: int) -> tuple[list, list[str]]:
        pool = list(self.pool)
        per = {m: self.p_map(a, b, m) for m in pool}
        comfort = {"A": lambda m: self._comfort(a, per[m], m),
                   "B": lambda m: self._comfort(b, 1 - per[m], m)}
        if best_of == 1:
            order = ["ban"] * 6
        elif best_of >= 5:
            order = ["ban", "ban", "pick", "pick", "pick", "pick"]
        else:
            order = ["ban", "ban", "pick", "pick", "ban", "ban"]
        veto, played = [], []
        for i, action in enumerate(order):
            if len(pool) <= 1:
                break
            who = "A" if i % 2 == 0 else "B"
            choose = min if action == "ban" else max
            m = choose(pool, key=comfort[who])
            pool.remove(m)
            veto.append((who, action, m))
            if action == "pick":
                played.append(m)
        if pool:
            veto.append(("-", "decider", pool[0]))
            played.append(pool[0])
        return veto, played

    @staticmethod
    def series(probs: list[float], best_of: int) -> tuple[float, dict[str, float]]:
        need = best_of // 2 + 1
        # если карт в маппуле не хватило, недостающие считаем «средними»
        probs = list(probs) + [sum(probs) / len(probs) if probs else 0.5] * (best_of - len(probs))
        scores: Counter = Counter()

        def walk(i: int, wa: int, wb: int, pr: float) -> None:
            if wa == need or wb == need:
                scores[f"{wa}:{wb}"] += pr
                return
            walk(i + 1, wa + 1, wb, pr * probs[i])
            walk(i + 1, wa, wb + 1, pr * (1 - probs[i]))

        walk(0, 0, 0, 1.0)
        p_a = sum(v for k, v in scores.items() if int(k.split(":")[0]) == need)
        return p_a, dict(sorted(scores.items(), key=lambda kv: -kv[1]))

    def predict(self, a: MapTeam, b: MapTeam, best_of: int = 3) -> MapPrediction:
        best_of = best_of if best_of in (1, 3, 5) else 3
        per = {m: self.p_map(a, b, m) for m in self.pool}
        veto, played = self.simulate_veto(a, b, best_of)
        probs = [per[m] for m in played][:best_of]
        p_a, scores = self.series(probs, best_of)
        return MapPrediction(self.pool, per, veto, played, p_a, scores)
