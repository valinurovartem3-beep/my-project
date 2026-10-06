"""Загрузка счёта по картам CS2 из Liquipedia (MediaWiki API, бесплатно).

Правила Liquipedia: не чаще 1 запроса в 2 секунды, обязательный User-Agent
с контактами, сжатие gzip, кеширование. Мы берём запас и делаем паузу
не меньше REQUEST_INTERVAL секунд. Данные распространяются по лицензии
CC-BY-SA 3.0, поэтому бот указывает Liquipedia как источник.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

API_URL = "https://liquipedia.net/counterstrike/api.php"
REQUEST_INTERVAL = 5.0
TIER_PAGES = {"S": "S-Tier_Tournaments", "A": "A-Tier_Tournaments", "B": "B-Tier_Tournaments"}


# ---------- разбор вики-разметки ----------

def _find_close(text: str, start: int) -> int:
    """Индекс сразу после '}}', закрывающего шаблон, начатый на позиции start."""
    depth, i, n = 0, start, len(text)
    while i < n - 1:
        pair = text[i:i + 2]
        if pair == "{{":
            depth += 1
            i += 2
        elif pair == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return n


def parse_template(src: str) -> tuple[str, dict[str, str], list[str]]:
    """'{{Name|a=1|b|{{X|y}}}}' -> ('Name', {'a': '1'}, ['b', ...])."""
    body = src[2:-2]
    parts, depth, buf, i = [], 0, [], 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; buf.append(two); i += 2; continue
        if two in ("}}", "]]"):
            depth -= 1; buf.append(two); i += 2; continue
        if body[i] == "|" and depth == 0:
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(body[i]); i += 1
    parts.append("".join(buf))
    name = parts[0].strip()
    named, positional = {}, []
    for p in parts[1:]:
        k, eq, v = p.partition("=")
        if eq and re.fullmatch(r"\s*[\w ]+\s*", k):
            named[k.strip()] = v.strip()
        else:
            positional.append(p.strip())
    return name, named, positional


def iter_templates(text: str, name: str):
    for m in re.finditer(r"\{\{\s*" + re.escape(name) + r"\s*(?=[|}\n])", text):
        end = _find_close(text, m.start())
        yield text[m.start():end]


TZ_OFFSETS = {"UTC": 0, "GMT": 0, "WET": 0, "CET": 1, "CEST": 2, "EET": 2, "EEST": 3, "MSK": 3,
              "BST": 1, "GST": 4, "IST": 5.5, "SGT": 8, "CST": 8, "KST": 9, "JST": 9, "AEST": 10,
              "AEDT": 11, "BRT": -3, "EST": -5, "EDT": -4, "CDT": -5, "PST": -8, "PDT": -7}


def parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    tz = re.search(r"Abbr/(\w+)", raw)
    offset = TZ_OFFSETS.get(tz.group(1).upper(), 0) if tz else 0
    clean = re.sub(r"\{\{.*?\}\}", "", raw).strip(" -")
    for fmt in ("%B %d, %Y - %H:%M", "%B %d, %Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(clean, fmt)
            return (dt - timedelta(hours=offset)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _int(v: str | None) -> int:
    try:
        return int(re.sub(r"[^\d]", "", v or "") or 0)
    except ValueError:
        return 0


def parse_maps(wikitext: str, page: str, tier: str = "") -> list[dict]:
    """Все сыгранные карты из матчей на странице турнира."""
    out = []
    for src in iter_templates(wikitext, "Match"):
        _, p, _ = parse_template(src)
        teams = []
        for key in ("opponent1", "opponent2"):
            _, _, pos = parse_template(p.get(key, "{{}}")) if p.get(key, "").startswith("{{") else ("", {}, [])
            teams.append((pos[0] if pos else "").strip().lower())
        if not all(teams) or "tbd" in teams:
            continue
        date = parse_date(p.get("date", ""))
        for i in range(1, 8):
            raw = p.get(f"map{i}", "")
            if not raw.startswith("{{"):
                continue
            _, mp, _ = parse_template(raw)
            name = mp.get("map", "").strip()
            if not name or mp.get("finished", "").lower() not in ("true", "1"):
                continue
            t1t, t1ct, t2t, t2ct = (_int(mp.get(k)) for k in ("t1t", "t1ct", "t2t", "t2ct"))
            ot1 = sum(_int(v) for k, v in mp.items() if re.fullmatch(r"o\d+t1(t|ct)", k))
            ot2 = sum(_int(v) for k, v in mp.items() if re.fullmatch(r"o\d+t2(t|ct)", k))
            s1, s2 = t1t + t1ct + ot1, t2t + t2ct + ot2
            if s1 == s2 == 0:  # старый формат: только итоговый счёт
                s1, s2 = _int(mp.get("score1")), _int(mp.get("score2"))
            if s1 == s2:
                continue
            out.append({
                "page": page, "tier": tier, "date": date.isoformat() if date else None,
                "map": name, "team1": teams[0], "team2": teams[1],
                "score1": s1, "score2": s2,
                # раунды по сторонам (без овертайма): [побед за T, раундов за T, побед за CT, раундов за CT]
                "t1sides": [t1t, t1t + t2ct, t1ct, t1ct + t2t],
                "t2sides": [t2t, t2t + t1ct, t2ct, t2ct + t1t],
            })
    return out


# ---------- клиент API ----------

class Liquipedia:
    def __init__(self, contact: str, cache_file: Path, tiers: str = "S,A",
                 history_days: int = 180):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": f"EsportsPredictBot/1.0 ({contact})",
                     "Accept-Encoding": "gzip"},
            timeout=30.0)
        self._last = 0.0
        self._lock = asyncio.Lock()
        self.cache_file = cache_file
        self.tiers = [t.strip().upper() for t in tiers.split(",") if t.strip()]
        self.history_days = history_days
        self.cache = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.cache_file.read_text())
        except (FileNotFoundError, ValueError):
            return {"subpages": {}, "pages": {}}

    def _save(self) -> None:
        self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False))

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, **params) -> dict:
        params.update(format="json", formatversion="2")
        async with self._lock:
            wait = REQUEST_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                r = await self._client.get(API_URL, params=params)
            finally:
                self._last = time.monotonic()
        if r.status_code == 429 or "Rate Limited" in r.text[:300]:
            raise RuntimeError("Liquipedia ограничила частоту запросов, обновление отложено")
        r.raise_for_status()
        return r.json()

    async def _tournament_list(self) -> list[tuple[str, str]]:
        """(страница, уровень) турниров за этот и прошлый год + текущие из Liquipedia:Tournaments."""
        year = datetime.now().year
        years = {str(year), str(year - 1)}
        found: dict[str, str] = {}
        for tier in self.tiers:
            d = await self._get(action="query", prop="links", titles=TIER_PAGES[tier],
                                pllimit="500", plnamespace="0")
            for link in d["query"]["pages"][0].get("links", []):
                t = link["title"]
                ys = set(re.findall(r"\b(20\d\d)\b", t))
                if ys & years and not any(int(y) > year for y in ys):
                    found.setdefault(t, tier)
        d = await self._get(action="query", prop="revisions", rvprop="content", rvslots="main",
                            titles="Liquipedia:Tournaments")
        text = d["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
        section = text.split("*Upcoming")[-1]
        section = section.split("*Ongoing", 1)[-1] if "*Ongoing" in section else section
        for line in section.splitlines():
            if line.startswith("**"):
                found.setdefault(line[2:].split("|")[0].replace("_", " ").strip(), "")
        return list(found.items())

    async def _subpages(self, title: str, refresh: bool) -> list[str]:
        cached = self.cache["subpages"].get(title)
        if cached is not None and not refresh:
            return cached
        d = await self._get(action="query", list="allpages", apprefix=title,
                            aplimit="100", apnamespace="0")
        pages = [p["title"] for p in d["query"]["allpages"]
                 if p["title"] == title or p["title"].startswith(title + "/")]
        self.cache["subpages"][title] = pages or [title]
        return self.cache["subpages"][title]

    async def update(self) -> int:
        """Обновляет кеш. Возвращает число сыгранных карт в окне истории."""
        tournaments = await self._tournament_list()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.history_days)
        all_pages: dict[str, str] = {}
        two_weeks_ago = datetime.now(timezone.utc) - timedelta(days=14)
        for title, tier in tournaments:
            known = [self.cache["pages"].get(sp) for sp in self.cache["subpages"].get(title, [])]
            dates = [i["last_date"] for i in known if i and i.get("last_date")]
            last = datetime.fromisoformat(max(dates)) if dates else None
            # турнир давно закончился и уже скачан — больше не трогаем
            if last and last < two_weeks_ago and all(known):
                continue
            for sp in await self._subpages(title, refresh=True):
                all_pages[sp] = tier

        titles = list(all_pages)
        for i in range(0, len(titles), 50):
            chunk = titles[i:i + 50]
            d = await self._get(action="query", prop="revisions", rvprop="ids",
                                titles="|".join(chunk))
            changed = [p["title"] for p in d["query"]["pages"]
                       if "revisions" in p and
                       self.cache["pages"].get(p["title"], {}).get("rev") != p["revisions"][0]["revid"]]
            for j in range(0, len(changed), 10):  # контент тяжёлый — по 10 страниц
                sub = changed[j:j + 10]
                d = await self._get(action="query", prop="revisions", rvprop="ids|content",
                                    rvslots="main", titles="|".join(sub))
                for p in d["query"]["pages"]:
                    if "revisions" not in p:
                        continue
                    rev = p["revisions"][0]
                    maps = parse_maps(rev["slots"]["main"]["content"], p["title"],
                                      all_pages.get(p["title"], ""))
                    dates = [m["date"] for m in maps if m["date"]]
                    last = max(dates) if dates else None
                    self.cache["pages"][p["title"]] = {
                        "rev": rev["revid"], "maps": maps, "last_date": last,
                        "stale": bool(last and datetime.fromisoformat(last) < cutoff),
                    }
            self._save()
        self._save()
        return len(self.maps())

    def maps(self) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.history_days)).isoformat()
        seen, out = set(), []
        for info in self.cache["pages"].values():
            for m in info.get("maps", []):
                key = (m["date"], m["team1"], m["team2"], m["map"], m["score1"], m["score2"])
                if m["date"] and m["date"] >= cutoff and key not in seen:
                    seen.add(key)
                    out.append(m)
        return sorted(out, key=lambda m: m["date"])
