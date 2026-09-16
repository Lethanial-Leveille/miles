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
- `miles-server.service` runs uvicorn server:app on port 8000 (FastAPI),
  bound to `localhost` rather than to an address
- `miles-tunnel.service` runs cloudflared tunnel (Cloudflare Tunnel)

`--host localhost` rather than `--host 127.0.0.1`, and the difference is not
cosmetic. asyncio resolves a hostname and binds a socket per result, so
`localhost` yields two listeners, `127.0.0.1:8000` and `[::1]:8000`, while the
literal address yields only the first. cloudflared routes to
`http://localhost:8000` and so may present either family. Confirm both after
any change to that line, because a single listener here fails only for whichever
family cloudflared happens to pick:

```bash
ss -ltnp | grep 8000        # expect a 127.0.0.1 line AND a [::1] line
```

It was `0.0.0.0` until Sep 15 2026. **The consequence is that nothing off the Pi
reaches the API directly any more.** The Nova iOS app must go through
miles.lethanial.com; talking to `miles.local:8000` or the Pi's LAN address now
gets a refused connection, and that is the intent rather than a fault.

Timer driven, not long running:

- `miles-health.timer` runs scripts/healthcheck.py every fifteen minutes
- `miles-wifi.timer` runs scripts/wifi_watchdog.sh every two minutes

`miles-wifi` checks whether wlan0 has a carrier and a global scope IPv4
address, and runs `nmcli connection up Alsander` when either is missing. It
runs as root because `nmcli connection up` is polkit protected and a non root
caller is refused on the one occasion it has to work. It is silent on a healthy
link, because systemd already writes a Starting and a Finished line per run and
seven hundred daily lines saying nothing happened would bury the ones that
matter:

```bash
systemctl list-timers miles-wifi.timer
journalctl -u miles-wifi.service -n 50
```

**Every unit is versioned under `systemd/`** and installed by copying to
`/etc/systemd/system/`. That became true on Sep 15 2026; before then only the
`miles-health` pair was in the repo and the three long running services existed
nowhere but `/etc`, which meant a reinstall rebuilt them from memory.

`systemd/` is a copy, not the running configuration, so the two can still
diverge. What keeps them honest is checking, not hoping:

```bash
for u in miles-voice miles-server miles-tunnel miles-health miles-wifi; do
    diff -q "/etc/systemd/system/$u.service" ~/miles/systemd/"$u.service"
done
```

Editing under `/etc` and forgetting to copy back is the drift this invites.
Change the repo copy first, then install it.

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

## SSH configuration

**Read the effective config with `sudo sshd -T`. Never by reading a file.**

```bash
sudo sshd -T | grep -iE "^(passwordauthentication|permitrootlogin|allowusers|pubkeyauthentication)\b"
```

Expected on this Pi, as of Sep 15 2026:

```
passwordauthentication no
permitrootlogin without-password
allowusers theycallmelee
pubkeyauthentication yes
```

The reason a file is not good enough is that the config is assembled from
several, and the assembly rule is the opposite of what almost everyone assumes.

`/etc/ssh/sshd_config` line 12 is:

```
Include /etc/ssh/sshd_config.d/*.conf
```

That include is at the **top** of the file, not the bottom, and **sshd takes
the first value it finds for any keyword and ignores every later one**. There is
no last write wins here and no override semantics. First occurrence is the
value, so:

- The drop in directory beats the main file, because it is included first.
- Within the directory, files are read in lexical order, so a **lower numbered
  file wins**.

Which makes the current state read backwards until you know the rule:

| File | Sets | Wins? |
|---|---|---|
| `00-hardening.conf` | `PasswordAuthentication no` | **yes**, it sorts first |
| `50-cloud-init.conf` | `PasswordAuthentication yes` | no, sshd already has a value |

`00-hardening.conf` is numbered the way it is for exactly one reason: to sort
ahead of `50-cloud-init.conf`. That file is **managed by cloud-init and may be
regenerated**, so editing it to say `no` is not a fix, it is a change waiting to
be reverted by something that does not know why it mattered. Winning on sort
order is durable in a way that editing a generated file is not.

`AllowUsers theycallmelee` is the exception to all of this: it lives in
`/etc/ssh/sshd_config` itself, at line 59, because nothing in the drop in
directory sets it and first occurrence therefore still finds it.

After any change, test before reloading, because a config that fails to parse
takes sshd down and this is a headless machine:

```bash
sudo sshd -t && sudo systemctl reload ssh     # -t validates, reload applies
sudo sshd -T | grep -i passwordauthentication # confirm what actually took
```

Keep a second session open while doing it. Why this is not a theoretical
concern is in [INCIDENTS.md](INCIDENTS.md#one-deauth-cost-21-hours-offline-and-9-hours-exposed-sep-14-2026).

## Tailscale

Third way to reach the Pi, added Sep 15 2026 after the incident below. The
other two are the LAN and the Cloudflare tunnel.

| Path | Reaches | Requires |
|---|---|---|
| LAN | `miles.local`, port 22 only | being on the same network |
| Cloudflare tunnel | miles.lethanial.com, the FastAPI app | nothing, it is public |
| Tailnet | `miles` at `100.99.248.127`, any port | being on the tailnet |

```bash
tailscale status          # who is on the tailnet
tailscale ip -4           # this node's address
tailscale ping <machine>  # round trip, and whether it is direct or relayed
```

**Why it is here.** The Sep 14 incident's recovery required a publicly
addressable machine, which is how three services ended up facing the internet
for nine hours. The tailnet makes the Pi reachable without any listener facing
the internet at all, so the next recovery does not have to trade an outage for
an exposure.

**Tailscale SSH is deliberately off.** `tailscale up` ran without `--ssh`, so
SSH is still governed by the hardened sshd config above rather than by tailnet
ACLs. Reaching the Pi over the tailnet still uses your key and still obeys
`AllowUsers`.

**MagicDNS rewrites `/etc/resolv.conf`** to `100.100.100.100` and adds your
tailnet's search domain. Name resolution was verified working through
it for the Anthropic, ElevenLabs and OpenWeatherMap hosts. If anything ever
starts reporting `no_dns`, this is the first place to look rather than the
router.

**The CGNAT ranges overlap, and today they do not collide.** Tailscale assigns
from `100.64.0.0/10` and this Pi's wlan0 lease is `100.70.16.218/25`, inside
that same range, with a link scope route for `100.70.16.128/25` out `wlan0`.
Both current nodes are clear of it, `miles` at `100.99.248.127` and the MacBook
at `100.103.30.91`. The failure to watch for is narrow and specific: a future
node assigned between `100.70.16.128` and `100.70.16.255` would be routed out
wlan0 from this Pi and be unreachable from here, and nowhere else.

```bash
ip route | grep 100.      # the local /25 that tailnet traffic must not land in
```

## FastAPI endpoints

REST: /auth/login, /auth/refresh, /chat, /chat/stream, /memories,
/memories/pending, /memories/{id}/approve, /memories/{id} (DELETE),
/history, /status, /docs
WebSocket: /ws

`/chat/stream` is the same turn as `/chat`, sent as Server Sent Events while it
is written: `event: delta` per piece, `event: reset` when what was streamed is
being dropped, then `event: done` with the finished reply, or `event: error`.
The turn runs on a worker thread, because `ask_nova` starts its own event loop,
and it finishes whether or not the client is still listening. A comment line
goes out every 15 seconds of silence so the tunnel does not close a connection
waiting on a slow tool.

**`/ws` is broken and has been since it was written.** The handler is `async`
and calls `ask_nova`, which calls `asyncio.run` inside the loop that is already
running, so every message raises. Nothing uses it: the app posts to `/chat`.
Found Sep 14 2026, left alone rather than fixed blind; see BACKEND_TODO.md.

Auth: JWT, HS256, Authorization Bearer header. Access tokens expire after 60
minutes (`ACCESS_TOKEN_EXPIRE_MINUTES` in `auth.py`). No refresh token is issued;
`/auth/refresh` needs a token that has not expired yet.

## Environment variables

The services read `~/miles/.env` (gitignored, mode 600) through systemd
`EnvironmentFile`, and `config.py` loads the same file for anything run by hand.
`~/.bashrc` also exports ANTHROPIC_API_KEY, WEATHER_API_KEY and FISH_API_KEY, but
only interactive shells see it; no service does.

`~/miles/.env`:
- MILES_PASSWORD_HASH
- MILES_JWT_SECRET
- ANTHROPIC_API_KEY
- WEATHER_API_KEY
- ELEVENLABS_API_KEY
- FISH_API_KEY (retained for rollback only)
- OURA_CLIENT_ID, OURA_CLIENT_SECRET

The voice id used to live here. It moved to config.py: a voice id is neither
secret nor deployment specific, and keeping it in a gitignored file meant voice
changes carried no history. `ELEVENLABS_VOICE_ID` is now unused and can be
deleted from .env.

### OAuth tokens

Two outside services hold a standing grant, each stored as a token file in
`data/`, gitignored and mode 600. Never give a credential a fallback value in
source: `oura_auth.py` briefly carried the real client secret as a default.

- `data/token.json`: Google Calendar. Created by `python3 scripts/google_auth.py`
  from `credentials.json`, the OAuth client, also gitignored and 600. Refreshed
  in memory on use and not written back.
- `data/oura_token.json`: Oura. Created by `python3 scripts/oura_auth.py`.
  Refreshed under a file lock and saved by write then rename, because voice and
  server are separate processes sharing one token.

**If the Google Cloud OAuth app is still in Testing, Google expires its refresh
tokens after seven days**, and every calendar tool starts failing with an auth
error. Moving the app to In production removes the limit; for a personal app the
unverified warning screen can be clicked through.

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
