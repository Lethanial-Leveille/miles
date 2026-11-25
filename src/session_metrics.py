"""The nightly review's real numbers: how often he finishes it, what each one
costs, and whether the must dos it planned got done.

Everything is read from rows the sessions already write; nothing extra is
logged at runtime. The arithmetic is pure functions over those rows so it is
tested without a database, and scripts/session_metrics.py prints it.

Definitions, chosen so a changelog line can quote them honestly:
- completed: he closed it himself (closed_by 'user').
- abandoned: he wrote in it but the 3am timer had to close it.
- missed: he never wrote in it.
- completion rate: completed over every settled review (open ones excluded).
- cost: every assistant turn plus the close summary call, at list prices in
  config.MODEL_PRICES_PER_MTOK. A model with no price makes the cost None,
  never a guess.
- must dos: those dated the day after the review, counted only once that day
  is over, done over planned.
"""

from config import MODEL_PRICES_PER_MTOK

COMPLETED, ABANDONED, MISSED, OPEN = "completed", "abandoned", "missed", "open"


def outcome(session, user_messages):
    if session["status"] == "open":
        return OPEN
    if session["status"] == "missed" or user_messages == 0:
        return MISSED
    return COMPLETED if session["closed_by"] == "user" else ABANDONED


def cost(usage, summary_model=None, summary_in=None, summary_out=None):
    """Dollars for one session. usage rows are (model, input, output, cache
    read, cache write, turns)."""
    total = 0.0
    lines = [(m, i or 0, o or 0, cr or 0, cw or 0) for m, i, o, cr, cw, _ in usage]
    if summary_model:
        lines.append((summary_model, summary_in or 0, summary_out or 0, 0, 0))
    for model, inp, out, cache_read, cache_write in lines:
        prices = MODEL_PRICES_PER_MTOK.get(model)
        if prices is None:
            return None
        p_in, p_out, p_read, p_write = prices
        total += (inp * p_in + out * p_out + cache_read * p_read + cache_write * p_write) / 1e6
    return total


def summarize(rows, today):
    """Per night lines and the totals, from database.session_metrics_rows.
    today is the current 4am day, so a must do day still in progress is not
    counted as missed work."""
    nights, settled, completed = [], 0, 0
    costs, planned, done = [], 0, 0
    for row in rows:
        s = row["session"]
        kind = outcome(s, row["user_messages"])
        dollars = cost(row["usage"], s.get("summary_model"),
                       s.get("summary_input_tokens"), s.get("summary_output_tokens"))
        if kind != OPEN:
            settled += 1
            completed += kind == COMPLETED
        if kind in (COMPLETED, ABANDONED) and dollars is not None:
            costs.append(dollars)
        day_over = row["must_dos_day"] < today
        if day_over and kind in (COMPLETED, ABANDONED):
            planned += row["must_dos_planned"]
            done += row["must_dos_done"]
        nights.append({"day": s["for_day"], "outcome": kind, "cost": dollars,
                       "must_dos_planned": row["must_dos_planned"],
                       "must_dos_done": row["must_dos_done"] if day_over else None})
    return {
        "nights": nights,
        "settled": settled,
        "completion_rate": completed / settled if settled else None,
        "average_cost": sum(costs) / len(costs) if costs else None,
        "must_dos_planned": planned,
        "must_dos_done": done,
        "must_do_rate": done / planned if planned else None,
    }
