"""Telegram-бот с прогнозами на матчи CS2 и Dota 2."""
from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)

from liquipedia import Liquipedia
from mapmodel import MapModel, MapTeam, norm
from model import EloModel, Prediction
from pandascore import GAMES, PandaScore, PandaScoreError
from tracker import Tracker, parse_ts

load_dotenv()
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("esports_bot")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
PANDASCORE_TOKEN = os.environ.get("PANDASCORE_TOKEN", "")
TZ = ZoneInfo(os.environ.get("TIMEZONE", "Europe/Moscow"))
HISTORY_PAGES = int(os.environ.get("HISTORY_PAGES", "10"))   # 10 стр. × 100 = 1000 матчей на игру
DIGEST_HOUR = int(os.environ.get("DIGEST_HOUR", "10"))
SUBS_FILE = Path(os.environ.get("SUBS_FILE", "subscribers.json"))
# Контакт для User-Agent — обязателен по правилам Liquipedia (почта или ник в Telegram)
LIQUIPEDIA_CONTACT = os.environ.get("LIQUIPEDIA_CONTACT", "")
LIQUIPEDIA_TIERS = os.environ.get("LIQUIPEDIA_TIERS", "S,A")
MAP_HISTORY_DAYS = int(os.environ.get("MAP_HISTORY_DAYS", "180"))

MODEL_REFRESH_SEC = 3 * 3600
MAPS_REFRESH_SEC = 6 * 3600
UPCOMING_TTL_SEC = 10 * 60
MATCHES_PER_PAGE = 8

api: PandaScore
models: dict[str, EloModel] = {g: EloModel() for g in GAMES}
upcoming_cache: dict[str, tuple[float, list[dict]]] = {}
liquipedia: Liquipedia | None = None
map_model = MapModel()
tracker: Tracker | None = None
TRACK_HOURS_AHEAD = 24  # автоматически записываем прогнозы на матчи ближайших суток

DISCLAIMER = "<i>Прогноз — статистическая оценка, а не гарантия результата.</i>"


# ---------- данные ----------

async def refresh_models(context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
    for game in GAMES:
        try:
            past = await api.past(game, pages=HISTORY_PAGES)
            models[game].fit(past)
            log.info("%s: модель обучена на %d матчах, команд: %d",
                     game, models[game].matches_used, len(models[game].teams))
        except Exception:
            log.exception("Не удалось обновить модель %s", game)


def fit_map_model() -> None:
    if liquipedia:
        map_model.fit(liquipedia.maps())
        log.info("Карты CS2: %d сыгранных карт, %d команд, маппул: %s",
                 map_model.maps_used, len(map_model.teams), ", ".join(map_model.pool))


async def refresh_maps(context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
    if not liquipedia:
        return
    try:
        await liquipedia.update()
    except Exception as e:
        log.warning("Liquipedia: обновление не удалось (%s), работаю на сохранённых данных", e)
    fit_map_model()


async def track_upcoming(context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
    """Записывает прогнозы на все матчи ближайших суток (обновляет, пока матч не начался)."""
    if not tracker:
        return
    now = datetime.now(TZ)
    n = 0
    for game in GAMES:
        if not models[game].teams:
            continue
        try:
            matches = await get_upcoming(game)
        except PandaScoreError:
            continue
        for m in matches:
            begin = parse_ts(m.get("begin_at") or m.get("scheduled_at"))
            if begin and 0 < (begin - now).total_seconds() < TRACK_HOURS_AHEAD * 3600:
                n += tracker.record(game, m, compute_prediction(game, m))
    log.info("Журнал: обновлено прогнозов — %d", n)


async def resolve_results(context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
    """Подтягивает результаты сыгранных матчей и отмечает, угадан ли прогноз."""
    if not tracker or not tracker.pending():
        return
    finished = {}
    for game in GAMES:
        try:
            for m in await api.past(game, pages=2):
                finished[(game, m["id"])] = m
        except PandaScoreError:
            log.warning("Не удалось получить результаты %s", game)
    n = tracker.resolve(finished)
    if n:
        log.info("Журнал: получены результаты %d матчей", n)


async def get_upcoming(game: str) -> list[dict]:
    cached = upcoming_cache.get(game)
    if cached and time.time() - cached[0] < UPCOMING_TTL_SEC:
        return cached[1]
    matches = [m for m in await api.upcoming(game, per_page=50)
               if len([o for o in m.get("opponents") or [] if o.get("opponent")]) == 2]
    upcoming_cache[game] = (time.time(), matches)
    return matches


def predict_match(game: str, m: dict) -> Prediction:
    a, b = (o["opponent"] for o in m["opponents"][:2])
    return models[game].predict(a, b, m.get("number_of_games"))


# ---------- форматирование ----------

def esc(s: str) -> str:
    return html.escape(s or "")


def fmt_time(m: dict) -> str:
    raw = m.get("begin_at") or m.get("scheduled_at")
    if not raw:
        return "время уточняется"
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(TZ)
    today = datetime.now(TZ).date()
    day = ("сегодня" if dt.date() == today else
           "завтра" if (dt.date() - today).days == 1 else dt.strftime("%d.%m"))
    return f"{day} {dt:%H:%M}"


def tournament_name(m: dict) -> str:
    league = (m.get("league") or {}).get("name") or ""
    serie = (m.get("serie") or {}).get("full_name") or ""
    tier = ((m.get("tournament") or {}).get("tier") or "").upper()
    name = " ".join(x for x in (league, serie) if x)
    return f"{name} [tier {tier}]" if tier else name


def bar(p: float, width: int = 10) -> str:
    n = round(p * width)
    return "█" * n + "░" * (width - n)


def short_line(game: str, m: dict) -> str:
    p = predict_match(game, m)
    fav, pf = (p.a, p.p_a) if p.p_a >= 0.5 else (p.b, 1 - p.p_a)
    bo = f"Bo{m['number_of_games']}" if m.get("number_of_games") else ""
    return (f"🕒 {fmt_time(m)} · {bo}\n"
            f"<b>{esc(p.a.name)}</b> vs <b>{esc(p.b.name)}</b>\n"
            f"➡️ {esc(fav.name)} — {pf:.0%}")


def full_card(game: str, m: dict) -> str:
    p = predict_match(game, m)
    g = GAMES[game]
    bo = f"Bo{m['number_of_games']}" if m.get("number_of_games") else "формат неизвестен"
    lines = [f"{g['emoji']} <b>{esc(p.a.name)} vs {esc(p.b.name)}</b>"]
    if m.get("_virtual"):
        lines.append(f"📌 Матча нет в ближайшем расписании — прогноз для {bo}")
    else:
        lines += [f"🏆 {esc(tournament_name(m) or 'турнир не указан')}",
                  f"🕒 {fmt_time(m)} (МСК) · {bo}"]
    lines += [
        "",
        "<b>Вероятность победы</b>",
        f"<code>{bar(p.p_a)}</code> {esc(p.a.name)} — <b>{p.p_a:.0%}</b>",
        f"<code>{bar(1 - p.p_a)}</code> {esc(p.b.name)} — <b>{1 - p.p_a:.0%}</b>",
        "",
        "<b>Команды</b>",
    ]
    for t in (p.a, p.b):
        form = t.form()
        form_s = f"{form:.0%}" if form is not None else "—"
        lines.append(f"• {esc(t.name)}: рейтинг {t.rating:.0f}, матчей {t.played}, "
                     f"форма {form_s}, последние: {t.streak() or '—'}")
    wa, wb = p.h2h
    lines.append(f"• Личные встречи (последние): {wa}–{wb}" if wa + wb else "• Личных встреч в базе нет")
    if p.factors:
        lines += ["", "<b>Ключевые факторы</b>"] + [f"• {esc(f)}" for f in p.factors]
    lines += ["", f"Уверенность модели: <b>{p.confidence}</b>", DISCLAIMER]
    return "\n".join(lines)


def pct(won: int, total: int) -> str:
    return f"{won / total:.0%}" if total else "—"


def maps_card(ta: MapTeam, tb: MapTeam, name_a: str, name_b: str, best_of: int) -> str:
    best_of = best_of if best_of in (1, 3, 5) else 3
    pr = map_model.predict(ta, tb, best_of)
    names = {"A": esc(name_a), "B": esc(name_b)}
    lines = [f"🗺 <b>{names['A']} vs {names['B']}</b> — прогноз по картам (Bo{best_of})", ""]

    lines.append("<b>Вероятный пик-бан</b> <i>(догадка модели)</i>")
    icons = {"ban": "❌", "pick": "✅", "decider": "⚖️"}
    verbs = {"ban": "банит", "pick": "выбирает"}
    for who, action, m in pr.veto:
        if action == "decider":
            lines.append(f"{icons[action]} {'Решающая' if best_of > 1 else 'Играют'}: <b>{esc(m)}</b>")
        else:
            lines.append(f"{icons[action]} {names[who]} {verbs[action]} {esc(m)}")

    lines += ["", f"<b>Шансы на каждой карте</b> ({names['A']} / {names['B']})"]
    for m in sorted(pr.pool, key=lambda x: -abs(pr.per_map[x] - 0.5)):
        p = pr.per_map[m]
        ra, rb = ta.maps.get(m), tb.maps.get(m)
        exp = (f"карт: {ra.played if ra else 0} / {rb.played if rb else 0}, "
               f"побед: {pct(ra.wins, ra.played) if ra else '—'} / {pct(rb.wins, rb.played) if rb else '—'}")
        mark = " ▸" if m in pr.played_maps else ""
        lines.append(f"<b>{esc(m)}</b>{mark} — {p:.0%} / {1 - p:.0%} <i>({exp})</i>")

    if best_of > 1:
        lines += ["", "<b>Счёт серии</b>"]
        for score, p in list(pr.scores.items())[:4]:
            lines.append(f"{score} — {p:.0%}")
    fav, pf = (names["A"], pr.series_p) if pr.series_p >= 0.5 else (names["B"], 1 - pr.series_p)
    lines += ["", f"➡️ По картам фаворит: <b>{fav}</b> — {pf:.0%}"]

    side_lines = []
    for m in pr.played_maps:
        ra, rb = ta.maps.get(m), tb.maps.get(m)
        if ra and rb and ra.t_total and rb.t_total:
            side_lines.append(
                f"{esc(m)}: {names['A']} T {pct(ra.t_won, ra.t_total)} / CT {pct(ra.ct_won, ra.ct_total)} · "
                f"{names['B']} T {pct(rb.t_won, rb.t_total)} / CT {pct(rb.ct_won, rb.ct_total)}")
    if side_lines:
        lines += ["", "<b>Раунды по сторонам</b> (доля выигранных)"] + side_lines

    warn = [n for n, t in ((names["A"], ta), (names["B"], tb)) if t.played < 10]
    if warn:
        lines += ["", f"⚠️ Мало сыгранных карт в базе: {', '.join(warn)} — прогноз ненадёжен"]
    lines += ["", "▸ — карты, которые, вероятно, будут сыграны",
              f"<i>Данные о картах: Liquipedia (CC-BY-SA), последние {MAP_HISTORY_DAYS} дней.</i>",
              DISCLAIMER]
    return "\n".join(lines)


def maps_for_match(m: dict) -> str:
    if not map_model.teams:
        return "Статистика по картам ещё загружается (первый раз это занимает несколько минут)."
    a, b = (o["opponent"] for o in m["opponents"][:2])
    ta = map_model.resolve(a.get("name", ""), a.get("acronym") or "")
    tb = map_model.resolve(b.get("name", ""), b.get("acronym") or "")
    missing = [x["name"] for x, t in ((a, ta), (b, tb)) if t is None]
    if missing:
        return (f"Нет данных по картам для: {esc(', '.join(missing))}. "
                f"Скорее всего, команда не играла турниры уровня {LIQUIPEDIA_TIERS} "
                f"за последние {MAP_HISTORY_DAYS} дней.")
    return maps_card(ta, tb, a["name"], b["name"], m.get("number_of_games") or 3)


def display_name(t: MapTeam) -> str:
    """Красивое название команды: из PandaScore, если совпадает, иначе код Liquipedia."""
    key = norm(t.slug)
    candidates = list(models["cs2"].teams.values())
    exact = [ps for ps in candidates if key in (norm(ps.name), norm(ps.acronym))]
    if exact:
        return max(exact, key=lambda ps: ps.played).name
    fuzzy = [ps for ps in candidates if map_model.resolve(ps.name, ps.acronym) is t]
    if fuzzy:
        return max(fuzzy, key=lambda ps: ps.played).name
    return t.slug.title() if t.slug.islower() else t.slug


def resolve_user_team(text: str) -> MapTeam | None:
    """Команда по тексту пользователя: напрямую или через название/аббревиатуру в PandaScore."""
    found = map_model.resolve(text)
    if found:
        return found
    key = norm(text)
    for ps in sorted(models["cs2"].teams.values(), key=lambda x: -x.played):
        if key and key in (norm(ps.name), norm(ps.acronym)):
            found = map_model.resolve(ps.name, ps.acronym)
            if found:
                return found
    return None


def match_buttons(game: str, m: dict) -> InlineKeyboardMarkup | None:
    if game == "cs2" and liquipedia:
        return InlineKeyboardMarkup([[InlineKeyboardButton(
            "🗺 Прогноз по картам", callback_data=f"mp:{m['id']}")]])
    return None


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔫 Матчи CS2", callback_data="list:cs2:0"),
         InlineKeyboardButton("🛡 Матчи Dota 2", callback_data="list:dota2:0")],
        [InlineKeyboardButton("📊 Рейтинг CS2", callback_data="top:cs2"),
         InlineKeyboardButton("📊 Рейтинг Dota 2", callback_data="top:dota2")],
        [InlineKeyboardButton("📈 Статистика прогнозов", callback_data="stats")],
    ])


# ---------- обработчики ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я анализирую матчи <b>CS2</b> и <b>Dota 2</b> и даю прогнозы "
        "на основе рейтинга команд, их текущей формы и личных встреч.\n\n"
        "👉 <b>Просто напишите матч</b>, например <code>NAVI vs Spirit</code>, — "
        "и я пришлю все прогнозы на него. Можно уточнить игру и формат: "
        "<code>Spirit vs Falcons дота bo3</code>\n\n"
        "Команды:\n"
        "/cs — ближайшие матчи CS2 с прогнозами\n"
        "/dota — ближайшие матчи Dota 2\n"
        "/maps NAVI vs Spirit — прогноз по картам CS2 (пик-бан, шансы на картах, счёт)\n"
        "/team &lt;название&gt; — карточка команды\n"
        "/top_cs, /top_dota — рейтинг команд\n"
        "/stats — сколько прогнозов бот угадал\n"
        "/subscribe — ежедневная сводка прогнозов\n"
        "/unsubscribe — отписаться",
        parse_mode=ParseMode.HTML, reply_markup=main_menu())


async def render_list(game: str, page: int) -> tuple[str, InlineKeyboardMarkup]:
    matches = await get_upcoming(game)
    g = GAMES[game]
    if not matches:
        return f"{g['emoji']} Ближайших матчей {g['title']} не найдено.", main_menu()
    chunk = matches[page * MATCHES_PER_PAGE:(page + 1) * MATCHES_PER_PAGE]
    text = f"{g['emoji']} <b>Ближайшие матчи {g['title']}</b>\n\n" + "\n\n".join(
        f"{i}. " + short_line(game, m) for i, m in enumerate(chunk, page * MATCHES_PER_PAGE + 1))
    text += "\n\nНажмите на матч, чтобы увидеть подробный разбор."
    buttons = [[InlineKeyboardButton(
        f"{i}. {m['opponents'][0]['opponent']['name']} vs {m['opponents'][1]['opponent']['name']}"[:60],
        callback_data=f"m:{game}:{m['id']}")]
        for i, m in enumerate(chunk, page * MATCHES_PER_PAGE + 1)]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️ Назад", callback_data=f"list:{game}:{page - 1}"))
    if (page + 1) * MATCHES_PER_PAGE < len(matches):
        nav.append(InlineKeyboardButton("Дальше ▶️", callback_data=f"list:{game}:{page + 1}"))
    if nav:
        buttons.append(nav)
    return text, InlineKeyboardMarkup(buttons)


def render_top(game: str) -> str:
    g = GAMES[game]
    top = models[game].top()
    if not top:
        return "Рейтинг ещё считается, попробуйте через минуту."
    rows = [f"{i}. {esc(t.name)} — {t.rating:.0f} ({t.streak()})" for i, t in enumerate(top, 1)]
    return f"📊 <b>Рейтинг команд {g['title']}</b> (по модели Эло)\n\n" + "\n".join(rows)


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE, game: str) -> None:
    try:
        text, kb = await render_list(game, 0)
    except PandaScoreError as e:
        text, kb = f"⚠️ {e}", None
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)


async def cmd_cs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_list(update, context, "cs2")


async def cmd_dota(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_list(update, context, "dota2")


async def cmd_top_cs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(render_top("cs2"), parse_mode=ParseMode.HTML)


async def cmd_top_dota(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(render_top("dota2"), parse_mode=ParseMode.HTML)


async def cmd_team(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Напишите название: /team NAVI")
        return
    blocks = []
    for game, g in GAMES.items():
        model = models[game]
        for t in model.find_team(query)[:2]:
            form = t.form()
            rank = next((i for i, x in enumerate(model.top(500, 8), 1) if x.id == t.id), None)
            recent = []
            for _, opp_id, win in reversed(t.recent[-5:]):
                opp = model.teams.get(opp_id)
                recent.append(f"{'✅' if win else '❌'} vs {esc(opp.name if opp else '?')}")
            blocks.append(
                f"{g['emoji']} <b>{esc(t.name)}</b> ({g['title']})\n"
                f"Рейтинг: {t.rating:.0f}" + (f" · место {rank}" if rank else "") + "\n"
                f"Матчей в базе: {t.played} · форма: {f'{form:.0%}' if form is not None else '—'}\n"
                + "\n".join(recent))
    await update.message.reply_text(
        "\n\n".join(blocks) if blocks else "Команда не найдена. Попробуйте другое написание.",
        parse_mode=ParseMode.HTML)


async def cmd_maps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    usage = "Формат: /maps NAVI vs Spirit (можно добавить bo1 или bo5 в конце)"
    if not liquipedia:
        await update.message.reply_text("Прогноз по картам выключен: не задан LIQUIPEDIA_CONTACT в .env.")
        return
    if not map_model.teams:
        await update.message.reply_text("Статистика по картам ещё загружается, попробуйте через несколько минут.")
        return
    text = " ".join(context.args)
    best_of = 3
    bo = re.search(r"\bbo\s*([135])\b", text, re.I)
    if bo:
        best_of = int(bo.group(1))
        text = text[:bo.start()] + text[bo.end():]
    parts = [p.strip() for p in re.split(r"\s+(?:vs|v|против|-)\s+", text, flags=re.I) if p.strip()]
    if len(parts) != 2:
        parts = text.split()
    if len(parts) != 2:
        await update.message.reply_text(usage)
        return
    ta, tb = resolve_user_team(parts[0]), resolve_user_team(parts[1])
    missing = [p for p, t in zip(parts, (ta, tb)) if t is None]
    if missing:
        await update.message.reply_text(
            f"Не нашёл в данных о картах: {', '.join(missing)}. Попробуйте другое написание. {usage}")
        return
    await update.message.reply_text(maps_card(ta, tb, display_name(ta), display_name(tb), best_of),
                                    parse_mode=ParseMode.HTML)


# ---------- статистика прогнозов ----------

def hit_rate(h: tuple[int, int]) -> str:
    hits, n = h
    return f"<b>{hits / n:.0%}</b> ({hits} из {n})" if n else "—"


def render_stats(days: int | None, game: str | None, asked_only: bool) -> str:
    rows = tracker.resolved(days, game, asked_only)
    scope = []
    if game:
        scope.append(GAMES[game]["title"])
    scope.append(f"за {days} дн." if days else "за всё время")
    if asked_only:
        scope.append("только ваши запросы")
    head = f"📈 <b>Статистика прогнозов</b> ({', '.join(scope)})"
    pending = tracker.db.execute(
        "SELECT COUNT(*) FROM predictions WHERE status='pending'").fetchone()[0]
    if not rows:
        return (f"{head}\n\nПока нет завершённых матчей с прогнозом. "
                f"Ожидают результата: {pending}.\n\n"
                "Бот сам записывает прогнозы на матчи ближайших суток и после матча сверяет результат — "
                "загляните сюда через день-два.")
    s = tracker.summarize(rows)
    by_game = {g: sum(r["game"] == g for r in rows) for g in GAMES}
    lines = [head, "",
             f"Матчей с результатом: <b>{s['n']}</b> (" +
             ", ".join(f"{GAMES[g]['title']}: {c}" for g, c in by_game.items() if c) + ")",
             f"Ожидают результата: {pending}", "",
             "<b>Угадан победитель</b>",
             f"• Итоговый прогноз: {hit_rate(s['p_final'])}",
             f"• Общий прогноз: {hit_rate(s['p_overall'])}"]
    if s["p_maps"][1]:
        lines.append(f"• Прогноз по картам (CS2): {hit_rate(s['p_maps'])}")
    if s["exact"][1]:
        lines.append(f"• Точный счёт серии: {hit_rate(s['exact'])}")

    lines += ["", "<b>Насколько можно верить процентам</b>",
              "<i>Если модель честна, при прогнозе ~65% угадывается ~65% матчей</i>"]
    for name, b in s["buckets"].items():
        if b.n:
            lines.append(f"• Прогноз {name}: угадано {b.hits / b.n:.0%} из {b.n} "
                         f"(модель обещала в среднем {b.p_sum / b.n:.0%})")
    if s["brier"] is not None:
        verdict = ("лучше монетки" if s["brier"] < 0.245 else
                   "на уровне монетки" if s["brier"] <= 0.255 else "хуже монетки")
        lines += ["", f"Индекс Брайера: <b>{format(s['brier'], '.3f').replace('.', ',')}</b> — {verdict} "
                      f"<i>(подбрасывание монетки = 0,250; чем меньше, тем лучше)</i>"]
    if s["n"] < 30:
        lines.append(f"⚠️ Матчей пока мало ({s['n']}) — проценты сильно зависят от случая. "
                     "Выводы надёжнее после 50–100 матчей.")

    lines += ["", "<b>Последние матчи</b>"]
    for r in s["last"]:
        p = r["p_final"]
        fav, pf = (r["team_a"], p) if p >= 0.5 else (r["team_b"], 1 - p)
        won = r["team_a"] if r["winner"] == "a" else r["team_b"]
        ok = "✅" if fav == won else "❌"
        exact = " 🎯" if r["top_score"] and r["top_score"] == r["score"] else ""
        lines.append(f"{ok} {esc(r['team_a'])} {r['score']} {esc(r['team_b'])} — "
                     f"ставил на {esc(fav)} ({pf:.0%}){exact}")
    lines += ["", "<i>Фильтры: /stats 7 · /stats 30 · /stats кс · /stats дота · /stats мои</i>"]
    return "\n".join(lines)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not tracker:
        await update.message.reply_text("Журнал прогнозов выключен.")
        return
    days, game, asked = None, None, False
    for arg in (a.lower() for a in context.args):
        if arg.isdigit():
            days = int(arg)
        elif GAME_HINTS["cs2"].fullmatch(arg):
            game = "cs2"
        elif GAME_HINTS["dota2"].fullmatch(arg):
            game = "dota2"
        elif arg in ("мои", "my", "mine"):
            asked = True
    await update.message.reply_text(render_stats(days, game, asked), parse_mode=ParseMode.HTML)


# ---------- анализ матча по тексту: «NAVI vs Spirit» ----------

GAME_HINTS = {
    "cs2": re.compile(r"(?<!\w)(cs2|cs|csgo|кс2|кс|контра)(?!\w)", re.I),
    "dota2": re.compile(r"(?<!\w)(dota2|dota|дота2|дота|доту)(?!\w)", re.I),
}
SEPARATOR = re.compile(r"\s+(?:vs\.?|v|x|против|—|–|-)\s+", re.I)


def parse_match_query(text: str) -> tuple[str | None, int | None, list[str] | None]:
    game = None
    for g, rx in GAME_HINTS.items():
        if rx.search(text):
            game = g
            text = rx.sub(" ", text)
    best_of = None
    bo = re.search(r"(?<!\w)(?:bo|бо)\s*([135])(?!\w)", text, re.I)
    if bo:
        best_of = int(bo.group(1))
        text = text[:bo.start()] + " " + text[bo.end():]
    text = text.strip()
    parts = [p.strip(" ,.!?") for p in SEPARATOR.split(text) if p and p.strip(" ,.!?")]
    if len(parts) != 2:
        dashed = [p.strip() for p in re.split(r"\s*[-–—]\s*", text) if p.strip()]
        words = text.split()
        parts = dashed if len(dashed) == 2 else words if len(words) == 2 else None
    return game, best_of, parts


def find_ps_team(game: str, query: str):
    key = norm(query)
    if not key:
        return None
    teams = list(models[game].teams.values())
    exact = [t for t in teams if key in (norm(t.name), norm(t.acronym))]
    if exact:
        return max(exact, key=lambda t: t.played)
    if len(key) >= 3:
        part = [t for t in teams if key in norm(t.name)]
        if part:
            return max(part, key=lambda t: t.played)
    return None


def as_opponent(t) -> dict:
    return {"opponent": {"id": t.id, "name": t.name, "acronym": t.acronym}}


async def find_scheduled(game: str, ta, tb) -> dict | None:
    try:
        matches = await get_upcoming(game)
    except PandaScoreError:
        return None
    for m in matches:
        ids = {o["opponent"]["id"] for o in m["opponents"][:2]}
        if ids == {ta.id, tb.id}:
            return m
    return None


def per_map_from_series(p_series: float, best_of: int) -> float:
    """Шанс выиграть одну карту/игру, при котором шанс выиграть серию равен p_series."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if MapModel.series([mid] * best_of, best_of)[0] < p_series:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def compute_prediction(game: str, m: dict) -> dict:
    """Все прогнозы на матч в одном месте — для сводки и для журнала."""
    p = predict_match(game, m)
    best_of = m.get("number_of_games") or 3
    best_of = best_of if best_of in (1, 2, 3, 5) else 3
    res = {"a": p.a.name, "b": p.b.name, "p_overall": p.p_a, "p_maps": None,
           "p_final": p.p_a, "top_score": None, "top_score_p": None, "confidence": p.confidence}
    scores = None
    if game == "cs2" and liquipedia and map_model.teams:
        mta = map_model.resolve(p.a.name, p.a.acronym)
        mtb = map_model.resolve(p.b.name, p.b.acronym)
        if mta and mtb and best_of != 2:
            mp = map_model.predict(mta, mtb, best_of)
            res["p_maps"] = mp.series_p
            res["p_final"] = (p.p_a + mp.series_p) / 2
            scores = mp.scores
    if scores is None and best_of in (3, 5):
        # для Dota 2 и команд без данных о картах — счёт из общего прогноза
        scores = MapModel.series([per_map_from_series(p.p_a, best_of)] * best_of, best_of)[1]
    if scores and best_of > 1:
        res["top_score"], res["top_score_p"] = next(iter(scores.items()))
    return res


async def analyze_match(game: str, ta, tb, best_of: int | None) -> list[tuple[str, InlineKeyboardMarkup | None]]:
    m = await find_scheduled(game, ta, tb)
    if m is None:
        m = {"_virtual": True, "id": 0, "number_of_games": best_of or 3,
             "opponents": [as_opponent(ta), as_opponent(tb)]}
    elif best_of:
        m = {**m, "number_of_games": best_of}

    pred = compute_prediction(game, m)
    if tracker and not m.get("_virtual"):
        tracker.record(game, m, pred, asked=True)
    a_name, b_name = pred["a"], pred["b"]
    g = GAMES[game]
    summary = [f"🎯 <b>{esc(a_name)} vs {esc(b_name)}</b> · {g['title']} — сводка", ""]
    summary.append(f"Общий прогноз (рейтинг, форма, личные встречи): "
                   f"{esc(a_name)} <b>{pred['p_overall']:.0%}</b> / {esc(b_name)} <b>{1 - pred['p_overall']:.0%}</b>")
    if pred["p_maps"] is not None:
        summary.append(f"Прогноз по картам: {esc(a_name)} <b>{pred['p_maps']:.0%}</b> / "
                       f"{esc(b_name)} <b>{1 - pred['p_maps']:.0%}</b>")
    if pred["top_score"]:
        summary.append(f"Самый вероятный счёт: <b>{pred['top_score']}</b> ({pred['top_score_p']:.0%})")
    if pred["p_maps"] is not None and abs(pred["p_overall"] - pred["p_maps"]) >= 0.15:
        summary.append("⚠️ Оценки заметно расходятся — исход особенно неочевиден")

    final = pred["p_final"]
    fav, pf = (a_name, final) if final >= 0.5 else (b_name, 1 - final)
    summary += ["", f"➡️ Итог: <b>{esc(fav)}</b> — {pf:.0%}",
                f"Уверенность: {pred['confidence']}"]
    if tracker and not m.get("_virtual"):
        summary.append("📝 Прогноз записан — после матча попадёт в /stats")
    summary += ["", "Подробности ниже 👇"]

    out = [("\n".join(summary), None), (full_card(game, m), None)]
    if game == "cs2" and liquipedia:
        out.append((maps_for_match(m), None))
    return out


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    game_hint, best_of, parts = parse_match_query(text)
    if not parts:
        await update.message.reply_text(
            "Напишите матч, например: <code>NAVI vs Spirit</code>\n"
            "Можно уточнить игру и формат: <code>Spirit vs Falcons дота bo3</code>",
            parse_mode=ParseMode.HTML)
        return
    if not any(models[g].teams for g in GAMES):
        await update.message.reply_text("Данные ещё загружаются, попробуйте через минуту.")
        return

    games = [game_hint] if game_hint else list(GAMES)
    found = {}
    for g in games:
        ta, tb = find_ps_team(g, parts[0]), find_ps_team(g, parts[1])
        if ta and tb and ta.id != tb.id:
            found[g] = (ta, tb)

    if not found:
        missing = [q for q in parts if not any(find_ps_team(g, q) for g in games)]
        if missing:
            msg = (f"Не нашёл команду: {esc(', '.join(missing))}. Попробуйте другое написание "
                   f"или полное название (например, <code>Natus Vincere</code> или <code>NAVI</code>).")
        else:
            msg = "Эти команды не играют в одну игру — проверьте названия или укажите игру (кс / дота)."
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    if len(found) > 1:
        # Команды есть в обеих играх (например, Spirit) — берём ту, где есть матч в расписании
        scheduled = [g for g, (ta, tb) in found.items() if await find_scheduled(g, ta, tb)]
        if len(scheduled) == 1:
            found = {scheduled[0]: found[scheduled[0]]}
        else:
            context.user_data["pending"] = {"parts": parts, "best_of": best_of}
            await update.message.reply_text(
                "Такие команды есть и в CS2, и в Dota 2. Какую игру разобрать?",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(f"{GAMES[g]['emoji']} {GAMES[g]['title']}", callback_data=f"an:{g}")
                    for g in found]]))
            return

    game, (ta, tb) = next(iter(found.items()))
    for msg, kb in await analyze_match(game, ta, tb, best_of):
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=kb)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    kind, *rest = q.data.split(":")
    try:
        if kind == "list":
            game, page = rest[0], int(rest[1])
            text, kb = await render_list(game, page)
            await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
        elif kind == "stats":
            text = render_stats(None, None, False) if tracker else "Журнал прогнозов выключен."
            await q.message.reply_text(text, parse_mode=ParseMode.HTML)
        elif kind == "top":
            await q.message.reply_text(render_top(rest[0]), parse_mode=ParseMode.HTML)
        elif kind == "m":
            game, mid = rest[0], int(rest[1])
            m = next((x for x in await get_upcoming(game) if x["id"] == mid), None)
            if m is None:
                await q.message.reply_text("Матч уже начался или пропал из расписания.")
                return
            if tracker:
                tracker.record(game, m, compute_prediction(game, m), asked=True)
            await q.message.reply_text(full_card(game, m), parse_mode=ParseMode.HTML,
                                       reply_markup=match_buttons(game, m))
        elif kind == "an":
            pending = context.user_data.pop("pending", None)
            if not pending:
                await q.message.reply_text("Напишите матч ещё раз, например: NAVI vs Spirit")
                return
            game = rest[0]
            ta, tb = (find_ps_team(game, x) for x in pending["parts"])
            for msg, kb in await analyze_match(game, ta, tb, pending["best_of"]):
                await q.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=kb)
        elif kind == "mp":
            mid = int(rest[0])
            m = next((x for x in await get_upcoming("cs2") if x["id"] == mid), None)
            if m is None:
                await q.message.reply_text("Матч уже начался или пропал из расписания.")
                return
            await q.message.reply_text(maps_for_match(m), parse_mode=ParseMode.HTML)
    except PandaScoreError as e:
        await q.message.reply_text(f"⚠️ {e}")


# ---------- подписка на ежедневную сводку ----------

def load_subs() -> set[int]:
    try:
        return set(json.loads(SUBS_FILE.read_text()))
    except (FileNotFoundError, ValueError):
        return set()


def save_subs(subs: set[int]) -> None:
    SUBS_FILE.write_text(json.dumps(sorted(subs)))


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subs = load_subs()
    subs.add(update.effective_chat.id)
    save_subs(subs)
    await update.message.reply_text(
        f"Готово! Каждый день в {DIGEST_HOUR:02d}:00 буду присылать прогнозы на матчи ближайших суток.")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subs = load_subs()
    subs.discard(update.effective_chat.id)
    save_subs(subs)
    await update.message.reply_text("Вы отписались от сводки.")


def build_digest(upcoming: dict[str, list[dict]]) -> str | None:
    now = datetime.now(TZ)
    parts = []
    for game, matches in upcoming.items():
        day = []
        for m in matches:
            raw = m.get("begin_at") or m.get("scheduled_at")
            if raw and (datetime.fromisoformat(raw.replace("Z", "+00:00")) - now).total_seconds() < 86400:
                day.append(m)
        if day:
            g = GAMES[game]
            parts.append(f"{g['emoji']} <b>{g['title']}</b>\n\n" +
                         "\n\n".join(short_line(game, m) for m in day[:10]))
    if tracker:
        rows = tracker.resolved(days=1)
        if rows:
            s = tracker.summarize(rows)
            hits, n = s["p_final"]
            parts.append(f"📈 За последние сутки угадано <b>{hits} из {n}</b> — подробнее /stats")
    if not parts:
        return None
    return "📅 <b>Прогнозы на ближайшие сутки</b>\n\n" + "\n\n".join(parts) + "\n\n" + DISCLAIMER


async def send_digest(context: ContextTypes.DEFAULT_TYPE) -> None:
    subs = load_subs()
    if not subs:
        return
    try:
        text = build_digest({g: await get_upcoming(g) for g in GAMES})
    except PandaScoreError:
        log.exception("Сводка не собрана")
        return
    if not text:
        return
    for chat_id in subs:
        try:
            await context.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML)
        except Exception:
            log.warning("Не удалось отправить сводку в %s", chat_id)


# ---------- запуск ----------

async def post_init(app: Application) -> None:
    await refresh_models()
    app.job_queue.run_repeating(refresh_models, interval=MODEL_REFRESH_SEC, first=MODEL_REFRESH_SEC)
    app.job_queue.run_daily(send_digest, time=dtime(hour=DIGEST_HOUR, tzinfo=TZ))
    if tracker:
        app.job_queue.run_repeating(track_upcoming, interval=30 * 60, first=60)
        app.job_queue.run_repeating(resolve_results, interval=60 * 60, first=120)
    if liquipedia:
        fit_map_model()  # сразу поднимаем то, что уже сохранено на диске
        # загрузка с Liquipedia идёт медленно (правила сайта), поэтому в фоне
        app.job_queue.run_repeating(refresh_maps, interval=MAPS_REFRESH_SEC, first=5)


async def post_shutdown(app: Application) -> None:
    await api.close()
    if liquipedia:
        await liquipedia.close()


def main() -> None:
    global api, liquipedia, tracker
    if not TELEGRAM_TOKEN or not PANDASCORE_TOKEN:
        raise SystemExit("Заполните TELEGRAM_TOKEN и PANDASCORE_TOKEN в файле .env (см. README.md)")
    api = PandaScore(PANDASCORE_TOKEN)
    tracker = Tracker(Path(os.environ.get("PREDICTIONS_DB", "predictions.db")))
    if LIQUIPEDIA_CONTACT:
        liquipedia = Liquipedia(LIQUIPEDIA_CONTACT, Path("liquipedia_cache.json"),
                                LIQUIPEDIA_TIERS, MAP_HISTORY_DAYS)
    else:
        log.warning("LIQUIPEDIA_CONTACT не задан — прогноз по картам отключён")
    app = (Application.builder().token(TELEGRAM_TOKEN)
           .post_init(post_init).post_shutdown(post_shutdown).build())
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("cs", cmd_cs))
    app.add_handler(CommandHandler("dota", cmd_dota))
    app.add_handler(CommandHandler("top_cs", cmd_top_cs))
    app.add_handler(CommandHandler("top_dota", cmd_top_dota))
    app.add_handler(CommandHandler("team", cmd_team))
    app.add_handler(CommandHandler("maps", cmd_maps))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
