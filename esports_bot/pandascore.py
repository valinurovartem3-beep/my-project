"""Клиент PandaScore API (бесплатный тариф: расписание и прошедшие матчи CS2 и Dota 2)."""
import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.pandascore.co"

# В PandaScore CS2 по-прежнему живёт под префиксом /csgo/
GAMES = {
    "cs2": {"slug": "csgo", "title": "CS2", "emoji": "🔫"},
    "dota2": {"slug": "dota2", "title": "Dota 2", "emoji": "🛡"},
}


class PandaScoreError(Exception):
    pass


class PandaScore:
    def __init__(self, token: str, timeout: float = 20.0):
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> list[dict]:
        for attempt in range(3):
            try:
                r = await self._client.get(path, params=params)
            except httpx.HTTPError as e:
                log.warning("PandaScore network error %s (attempt %d)", e, attempt + 1)
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429:  # лимит запросов
                await asyncio.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise PandaScoreError("Неверный или отсутствующий токен PandaScore")
            r.raise_for_status()
            return r.json()
        raise PandaScoreError("PandaScore не отвечает, попробуйте позже")

    async def upcoming(self, game: str, per_page: int = 50) -> list[dict]:
        slug = GAMES[game]["slug"]
        return await self._get(
            f"/{slug}/matches/upcoming",
            {"sort": "begin_at", "per_page": per_page, "page": 1},
        )

    async def running(self, game: str) -> list[dict]:
        slug = GAMES[game]["slug"]
        return await self._get(f"/{slug}/matches/running", {"per_page": 50})

    async def past(self, game: str, pages: int = 10, per_page: int = 100) -> list[dict]:
        """Последние завершённые матчи (до pages*per_page штук), от новых к старым."""
        slug = GAMES[game]["slug"]
        out: list[dict] = []
        for page in range(1, pages + 1):
            batch = await self._get(
                f"/{slug}/matches/past",
                {"sort": "-end_at", "per_page": per_page, "page": page,
                 "filter[status]": "finished"},
            )
            out.extend(batch)
            if len(batch) < per_page:
                break
        return out
