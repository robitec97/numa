"""Calendario dei concorsi (ora di Roma).

Dal luglio 2023 i concorsi si tengono martedi', giovedi', venerdi' e sabato
alle 20:00; la raccolta delle giocate chiude alle 19:30. Le festivita' a
volte spostano un concorso (per esempio al lunedi'): il registro delle
previsioni quindi non si fida della data prevista, ma abbina ogni previsione
alla prima estrazione avvenuta dopo il momento in cui e' stata registrata.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ROME = ZoneInfo("Europe/Rome")
DRAW_WEEKDAYS = (1, 3, 4, 5)       # martedi', giovedi', venerdi', sabato
DRAW_TIME = time(20, 0)
SALES_CLOSE = time(19, 30)
WEEKDAYS_IT = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")
MONTHS_IT = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
             "agosto", "settembre", "ottobre", "novembre", "dicembre")


def draw_moment(d: date) -> datetime:
    """Istante dell'estrazione del giorno d (anche per concorsi spostati)."""
    return datetime.combine(d, DRAW_TIME, ROME)


def next_draw_date(now: datetime) -> date:
    """Prossimo concorso in calendario per cui si puo' ancora giocare."""
    local = now.astimezone(ROME)
    d = local.date()
    while True:
        if d.weekday() in DRAW_WEEKDAYS and local < datetime.combine(d, SALES_CLOSE, ROME):
            return d
        d += timedelta(days=1)
        local = datetime.combine(d, time(0, 0), ROME)


def italian_date(d: date) -> str:
    return f"{WEEKDAYS_IT[d.weekday()]} {d.day} {MONTHS_IT[d.month - 1]} {d.year}"


def last_scheduled_draw(now: datetime) -> date:
    """Ultimo concorso in calendario gia' estratto da almeno un'ora."""
    local = now.astimezone(ROME)
    d = local.date()
    while not (d.weekday() in DRAW_WEEKDAYS and draw_moment(d) + timedelta(hours=1) <= local):
        d -= timedelta(days=1)
    return d
