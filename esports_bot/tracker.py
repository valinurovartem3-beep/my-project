"""Журнал прогнозов: записываем до начала матча, сверяем с результатом после.

Прогноз можно обновлять, пока матч не начался; после начала он «замораживается»,
поэтому статистика честная — модель не может подправить прогноз задним числом.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    match_id   INTEGER NOT NULL,
    game       TEXT NOT NULL,
    team_a     TEXT, team_b TEXT,
    team_a_id  INTEGER, team_b_id INTEGER,
    best_of    INTEGER,
    begin_at   TEXT,
    tournament TEXT,
    p_overall  REAL,            -- общий прогноз: вероятность победы команды A
    p_maps     REAL,            -- прогноз по картам (только CS2)
    p_final    REAL,            -- итоговый
    top_score  TEXT,            -- самый вероятный счёт, например «2:1»
    confidence TEXT,
    asked      INTEGER DEFAULT 0,  -- 1, если матч спрашивал пользователь
    updated_at TEXT,
    status     TEXT DEFAULT 'pending',  -- pending / resolved / void
    winner     TEXT,            -- 'a' или 'b'
    score      TEXT,            -- фактический счёт «a:b»
    PRIMARY KEY (match_id, game)
);
"""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


@dataclass
class Bucket:
    n: int = 0
    hits: int = 0
    p_sum: float = 0.0

    def add(self, hit: bool, p_fav: float) -> None:
        self.n += 1
        self.hits += hit
        self.p_sum += p_fav


class Tracker:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---------- запись ----------

    def record(self, game: str, m: dict, pred: dict, asked: bool = False) -> bool:
        """Сохраняет/обновляет прогноз. Возвращает False, если матч уже начался."""
        begin = parse_ts(m.get("begin_at") or m.get("scheduled_at"))
        if begin is None or begin <= now_utc() or m.get("_virtual"):
            return False
        a, b = (o["opponent"] for o in m["opponents"][:2])
        league = (m.get("league") or {}).get("name") or ""
        serie = (m.get("serie") or {}).get("full_name") or ""
        self.db.execute(
            """INSERT INTO predictions (match_id, game, team_a, team_b, team_a_id, team_b_id, best_of,
                   begin_at, tournament, p_overall, p_maps, p_final, top_score, confidence, asked, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(match_id, game) DO UPDATE SET
                   best_of=excluded.best_of, begin_at=excluded.begin_at, tournament=excluded.tournament,
                   p_overall=excluded.p_overall, p_maps=excluded.p_maps, p_final=excluded.p_final,
                   top_score=excluded.top_score, confidence=excluded.confidence,
                   asked=MAX(asked, excluded.asked), updated_at=excluded.updated_at
               WHERE predictions.status = 'pending' AND predictions.begin_at > excluded.updated_at""",
            (m["id"], game, a["name"], b["name"], a["id"], b["id"], m.get("number_of_games"),
             begin.isoformat(), f"{league} {serie}".strip(), pred["p_overall"], pred.get("p_maps"),
             pred["p_final"], pred.get("top_score"), pred.get("confidence"), int(asked),
             now_utc().isoformat()))
        self.db.commit()
        return True

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM predictions WHERE status='pending' AND begin_at < ?",
            (now_utc().isoformat(),)).fetchall()

    def resolve(self, finished: dict[tuple[str, int], dict]) -> int:
        """finished: (игра, id матча) -> матч PandaScore с результатом."""
        done = 0
        for row in self.pending():
            m = finished.get((row["game"], row["match_id"]))
            begin = parse_ts(row["begin_at"])
            if m is None:
                # матч отменили или перенесли без нового id
                if begin and now_utc() - begin > timedelta(days=3):
                    self.db.execute("UPDATE predictions SET status='void' WHERE match_id=? AND game=?",
                                    (row["match_id"], row["game"]))
                continue
            if m.get("forfeit") or m.get("winner_id") not in (row["team_a_id"], row["team_b_id"]):
                self.db.execute("UPDATE predictions SET status='void' WHERE match_id=? AND game=?",
                                (row["match_id"], row["game"]))
                continue
            scores = {r.get("team_id"): r.get("score") or 0 for r in m.get("results") or []}
            winner = "a" if m["winner_id"] == row["team_a_id"] else "b"
            score = f"{scores.get(row['team_a_id'], 0)}:{scores.get(row['team_b_id'], 0)}"
            self.db.execute(
                "UPDATE predictions SET status='resolved', winner=?, score=? WHERE match_id=? AND game=?",
                (winner, score, row["match_id"], row["game"]))
            done += 1
        self.db.commit()
        return done

    # ---------- статистика ----------

    def resolved(self, days: int | None = None, game: str | None = None,
                 asked_only: bool = False) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM predictions WHERE status='resolved'", []
        if days:
            q += " AND begin_at >= ?"
            args.append((now_utc() - timedelta(days=days)).isoformat())
        if game:
            q += " AND game = ?"
            args.append(game)
        if asked_only:
            q += " AND asked = 1"
        return self.db.execute(q + " ORDER BY begin_at DESC", args).fetchall()

    @staticmethod
    def summarize(rows: list[sqlite3.Row]) -> dict:
        def hit(p: float | None, winner: str) -> bool | None:
            if p is None or abs(p - 0.5) < 1e-9:
                return None
            return (p > 0.5) == (winner == "a")

        out = {"n": len(rows)}
        for key in ("p_final", "p_overall", "p_maps"):
            hits = [hit(r[key], r["winner"]) for r in rows]
            hits = [h for h in hits if h is not None]
            out[key] = (sum(hits), len(hits))

        brier = [((r["p_final"]) - (r["winner"] == "a")) ** 2 for r in rows]
        out["brier"] = sum(brier) / len(brier) if brier else None
        ll = [-math.log(max(1e-6, r["p_final"] if r["winner"] == "a" else 1 - r["p_final"])) for r in rows]
        out["logloss"] = sum(ll) / len(ll) if ll else None

        scored = [r for r in rows if r["top_score"] and r["score"]]
        out["exact"] = (sum(r["top_score"] == r["score"] for r in scored), len(scored))

        buckets = {"50–60%": Bucket(), "60–70%": Bucket(), "70–80%": Bucket(), "80%+": Bucket()}
        for r in rows:
            p_fav = max(r["p_final"], 1 - r["p_final"])
            h = hit(r["p_final"], r["winner"])
            if h is None:
                continue
            name = ("50–60%" if p_fav < 0.6 else "60–70%" if p_fav < 0.7 else
                    "70–80%" if p_fav < 0.8 else "80%+")
            buckets[name].add(h, p_fav)
        out["buckets"] = buckets
        out["last"] = rows[:10]
        return out
