# Infrastructure: services, tunnel, API, reminders

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Service and tunnel layout verified Aug 11 2026. The reminder poller was
> changed Sep 6 2026.
>
> **Live config values are declared in [CLAUDE.md](../CLAUDE.md), not here.**
> This file carries the reasoning and the measurements behind them. A number
> quoted inside a measurement is what was recorded on the date given and is
> left exactly as measured, so it may differ from what is live now. The config
> table in CLAUDE.md is the one that tracks the present.

Up: [CLAUDE.md](../CLAUDE.md) · [SESSION_START.md](SESSION_START.md) ·
[BACKEND_TODO.md](BACKEND_TODO.md) · [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) ·
[LATENCY.md](LATENCY.md) · [VOICE_OUTPUT.md](VOICE_OUTPUT.md) ·
[BRAIN.md](BRAIN.md) · [INFRASTRUCTURE.md](INFRASTRUCTURE.md) ·
[INCIDENTS.md](INCIDENTS.md)

---

What runs as a process, what is reachable from outside the Pi, and the one
piece of scheduled work the system owns.

## systemd services

- `miles-voice.service` runs voice_main.py (room mic pipeline)
- `miles-server.service` runs uvicorn server:app on port 8000 (FastAPI)
- `miles-tunnel.service` runs cloudflared tunnel (Cloudflare Tunnel)

```bash
sudo systemctl status miles-voice miles-server miles-tunnel
sudo systemctl restart miles-voice
journalctl -u miles-voice -f
journalctl -u miles-voice -n 50
```

The everyday commands live in [CLAUDE.md](../CLAUDE.md) so they are in front
of you without opening this file.

## Cloudflare tunnel

Tunnel name: miles. Routing changes happen at https://one.dash.cloudflare.com
under Networks > Tunnels > miles > Public Hostnames.

Active routes:
- miles.lethanial.com to http://localhost:8000 (primary)
- api.lethanial.com to http://localhost:8000 (legacy fallback)

Domain lethanial.com registered through Cloudflare. Free Zero Trust tier.

## FastAPI endpoints

REST: /auth/login, /auth/refresh, /chat, /memories, /memories/{id}, /history,
/status, /docs
WebSocket: /ws

Auth: JWT, HS256, Authorization Bearer header. Access tokens 7 day expiry.

## Environment variables

`~/.bashrc`:
- ANTHROPIC_API_KEY
- WEATHER_API_KEY
- FISH_API_KEY (retained for rollback only)

`~/miles/.env` (gitignored):
- MILES_PASSWORD_HASH
- MILES_JWT_SECRET
- ELEVENLABS_API_KEY

The voice id used to live here. It moved to config.py: a voice id is neither
secret nor deployment specific, and keeping it in a gitignored file meant voice
changes carried no history. `ELEVENLABS_VOICE_ID` is now unused and can be
deleted from .env.

## Reminders are fired by a poller, not by a thread

Changed Sep 6 2026. `set_reminder` used to write a row and then spawn a
`threading.Thread` that slept until the due time and fired from there.

**That made the thread the state and the row a record of it.** The row survived
a restart and the thread did not, and nothing scanned the table at boot, so
every pending reminder was silently dropped by any deploy or crash. With
`Restart=always` on the unit, both are routine. A reminder set for tomorrow
morning simply never happened, and the row sat at `completed = 0` forever with
nothing to distinguish it from one still waiting.

It had not bitten yet only because every reminder ever set was a one minute
test that fired before anything restarted.

`actions.poll_reminders` now reads `reminders` every `REMINDER_POLL_S` (20s) and
fires whatever is due. **There is no boot rearm, which is the point:** nothing is
held in memory, so a restart is just the next poll. `start_reminder_poller` is
called once from `voice_main.py`.

**Only the voice loop polls.** Both processes can create reminders and exactly
one may deliver them. A poller in the server would race for the same rows, and
the claim below would keep that correct while the alert landed in a queue
nothing drains.

**This fixed a second bug for free.** A reminder set through the app used to
spawn its thread inside the uvicorn process, so the alert queued into *that*
process's `alerts._pending`. `server.py` never calls `take_for_speech`, and
`brain.py` only folds, so it was delivered only if another chat message arrived
inside the 15 second fold window, and otherwise lost without even reaching
`alert_log`. Under the poller the row is created by whichever process and fired
by the voice loop.

**Completion happens before the alert is queued, deliberately.**
`complete_reminder` returns whether it changed a row, so the UPDATE doubles as a
claim and two passes cannot both win the same reminder. Firing first would be at
least once, whose failure mode is announcing every twenty seconds forever if
completion keeps failing. Claiming first is at most once, whose failure mode is
losing one reminder if the process dies in the microseconds between the commit
and the in memory append. The second is rarer and far less bad.

**Completion is by id.** The old code matched on `content AND due_at`, so two
reminders agreeing on both were closed by a single firing and only one was ever
spoken.

**A due time in the past is stored and announced, not dropped.** It is a bug
when it happens, almost always the clock guidance being ignored, and `alerts.py`
argues that silent non delivery is the worst available outcome. Past
`REMINDER_LATE_S` (1 hour) the wording says it came due while he was away,
because delivering a four hour old reminder as though it had just fired makes
the clock look broken.

**Timers are still in memory threads and do not survive a restart.** They are
not persisted at all, so there is no table and no record one ever existed.
Making them durable is a separate decision, not an oversight.

`REMINDER_POLL_S` and `REMINDER_LATE_S` are defined in `src/actions.py`, not
in `config.py`.

## Related

- Why a failed turn must not kill the voice loop, and why the boundary is not in `brain.py`: [BRAIN.md](BRAIN.md#failure-boundaries)
- What broke and how it was found: [INCIDENTS.md](INCIDENTS.md)
