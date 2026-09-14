# Voice output: synthesis, phrase bank, pronunciation

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Voice settings tuned by ear through Aug 12 2026. Phrase bank timings
> measured Aug 12 2026. Pronunciation model comparison measured Aug 11 2026.
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

How Nova sounds, and the three separate paths that produce audio: live
ElevenLabs synthesis, the pre rendered phrase bank, and the wake chime.

## Voice settings

The live model and settings are declared in CLAUDE.md. This section is why.

### Moved to eleven_v3, Sep 13 2026

On flash_v2 every reply sounded like a narrator stringing words together, "reading
and not speaking". Each suspect was rendered side by side on the same real reply
at a fixed seed and judged by ear:

| Changed | Heard a difference? |
|---|---|
| Sentences sent separately against the whole reply at once | No |
| Each sentence told its neighbours (`previous_text`, `next_text`) | No |
| Conversational wording against Nova's clipped sentences | Slightly |
| Speed 1.00 against 0.92 | No |
| Four other voices built for conversation | Victoria preferred |
| flash_v2 stability 0.90, 0.75, 0.60 | No |
| **eleven_v3** | **Yes, clearly** |

Measured on that reply: splitting it into sentences put about 300ms between each,
over 200ms of it waiting on ElevenLabs, and conversational wording cut the silence
inside the audio from 1.28s to 0.19s. Real, and not what he was hearing.

On v3, stability 0.5 sounded natural but a little exaggerated and slow. 1.0 kept
the delivery with less of both and ran shorter, 12.56s against 13.44s. v3
**ignored speed**: 1.0 and 1.1 rendered to identical lengths at both stabilities.
First audio was roughly 470 to 620ms against 380ms on flash, from single renders;
re measure in `timing_log` before quoting it.

Victoria herself is listed by ElevenLabs as `narrative_story`: "VO for explainer
videos, viral social media and modern brand ads. Warm, upbeat". He kept her
anyway, and the upbeat edge is partly the voice.

### One delivery for the rest of a reply, with a breath between sentences

Also Sep 13 2026, after the move to v3. He heard replies as "different cadences
that didn't match, like it could have been a different person", and as having
too few pauses with no variety in tone.

Each sentence had been its own request, and on v3 each request is voiced on its
own. Unseeded renders of one real reply, each played with no gap: every sentence
alone, the first alone and the rest together, and the whole reply at once. He
preferred the whole reply, and heard little difference from first plus rest.
The whole reply would mean waiting for Claude to finish before any sound, so
**the first sentence is sent at once and the rest together**, which gets the
single delivery without the wait. Giving each sentence its neighbours as context
was the other candidate, and ElevenLabs refuses it on v3: "Providing
previous_text or next_text is not yet supported with the 'eleven_v3' model."

For the pauses, the same whole reply at one seed, measured:

| Variant | Longest pause | Heard |
|---|---|---|
| stability 1.0 (live) | 210ms | too even |
| stability 0.75 | 200ms, and exactly the same length as 1.0 | v3 appears to treat it as 1.0 |
| stability 1.0, a full stop written as an ellipsis | **550ms** | chosen |
| stability 0.5 | 290ms | more variety, but the exaggeration he moved away from |

`tts.join_for_speech` writes the full stop between sentences as an ellipsis,
leaves a question or an exclamation alone, and shapes only what goes to
ElevenLabs. Commas are untouched; nothing tested gave them a longer pause.

### The flash_v2 tuning, kept as history

`VOICE_WITTY` and `VOICE_SERIOUS` are also defined and currently inert:
`speak()` falls back to `TTS_VOICE_SETTINGS` unless a caller passes an override,
and nothing selects one yet.

`stability` was tuned by ear, not chosen: 0.45 no, 0.60 eh, 0.75 pretty good,
0.90 good, 1.00 good, then the band between 0.75 and 0.90 searched directly.
0.60 was live during the whole period the voice sounded inconsistent.

Raised to 0.90 on Aug 12 2026 from live use: 0.80 read as too expressive and was
mispronouncing his name, which is the exact pair 0.90 is documented to address.

**The phrase bank renders at whatever this is.** Changing it means
`render_phrases.py render --force` over all of data/phrases, or the cached clips
and live speech drift apart. It also discards any variant already auditioned, so
re-pick those after.

It is one dial with two failure modes. Stability buys consistency by reducing
variation, and that same variation is what makes delivery sound alive, so 0.75
read better but occasionally missed the name and 0.90 held the name but read
flatter. Change it only after listening.

`use_speaker_boost` was believed unsupported on eleven_v3. **Corrected Sep 13 2026:** the API accepted it, and stability 0.9, on v3. Neither is a hard limit; whether v3 does anything with them is unmeasured.

## Phrase bank

Rendered Victoria played by aplay. `TTS_OUTPUT_FORMAT` is `pcm_22050` and the
chime is 22050Hz mono 16 bit, so the two are the same format and nothing
resamples.

Measured Aug 12 2026, after trimming, blocking wall time through the same aplay
path: chime 391ms, shortest ack ("Yep?", 0.358s) 388ms. **Indistinguishable in
mechanism, but not free in production**: the chime is fire and forget and
overlaps recording, while an ack must block, so an ack costs roughly 390ms that
the chime costs nothing.

An ack cannot overlap capture. webrtcvad at mode 2 does not read a tone as
speech, which is why the chime is safe, but an ack is speech: overlapped, it
endpoints the recording on Nova's own voice and lands at the head of the clip
Resemblyzer scores against his voiceprint.

**`play()` returns the text it played, not a bool.** The caller writes that to
history, and a bool meant the caller had to guess which variant ran. It guessed
index zero, so history could record "Goodnight, Lethanial." on a turn where she
actually said "Talk soon."

**`NIGHT_ONLY` filters variants by clock.** A conversation ends at any hour, and
"goodnight" at two in the afternoon is worse than no farewell. Kept as an index
filter rather than a separate key so a variant already auditioned and committed
keeps its file.

**Pinning the render seed freezes bad draws.** CLAUDE.md already notes that at
`stability = 0.80` the name is the unstable part: 0.75 "occasionally missed the
name", 0.90 "held the name but read flatter". Live, a bad reading of it passes
in one turn. Rendered, it is permanent. Use `render_phrases.py audition <key>
<index>` then `pick` for anything containing his name. Auditioning makes the
bank strictly better than live synthesis here, because live re-rolls the name
every time and can always draw badly.

ElevenLabs pads each clip with dead air, up to 232ms measured. `render_phrases.py
trim` strips it in place with no API call, kept separate from `render` because
ElevenLabs treats `seed` as best effort, so re-rendering risks redrawing the voice.

### Tuned from live use, Aug 12 2026

**`ACK_SPOKEN_CHANCE` 0.5 to 0.75**, and **`FOLLOWUP_TIMEOUT` 2.5 to 3.0**.

> **Correction, Sep 13 2026.** `ACK_SPOKEN_CHANCE` is `0.75` in source, which
> matches the change recorded here. The config list in CLAUDE.md still said
> `0.5` until today and has been corrected against source.

**Timer alerts said "your 5 minutes timer is up".** `_plural` is right for
"Timer set for five minutes", a quantity, and wrong before a noun, where the
unit is a modifier and must be singular. `_attributive` handles that role and
`_spoken_amount` spells the digit. Two grammatical roles, two helpers.

**Farewells now answer what he said.** `THANKS_ONLY` restricts "Anytime." and
"Any time at all." to dismissals that actually thanked her. "Never mind" is a
retraction, and answering it with "Any time at all" reads as not having
listened. Same index filter mechanism as `NIGHT_ONLY`, chosen so already
rendered files keep their numbering.

## Pronunciation

`pronunciations` table, one row per grapheme, seeded with Lethanial to
Luhthanyul. `tts.normalize_pronunciation` runs inside `speak()`, immediately
before the ElevenLabs call, so an alias reaches the synthesizer and nothing
else. Whole word, case insensitive, longest grapheme first.

`database.upsert_pronunciation()` adds entries at runtime with no migration.

`arpabet` is usable once `TTS_PHONEME_TAGS` is on, which requires a model that
honors phoneme tags. Measured Aug 11 2026: flash v2 honors them, flash v2.5
**drops the tagged word entirely**. Plain "Lethanial" gave 0.79s of audio;
wrapped in a phoneme tag on v2.5 it gave 0.23s, and deliberately absurd
phonemes gave the same 0.23s. On v2, correct phonemes matched plain at 0.74s
and absurd phonemes stretched to 1.21s.

Model is `eleven_flash_v2` for that reason. The switch is free: median time to
first byte over five runs was 349ms on v2 against 347ms on v2.5.

> **Correction, Sep 13 2026.** `TTS_PHONEME_TAGS` is `True` in source. The
> paragraph above reads as though it were still off. Arpabet is therefore
> live, not merely usable, and the flash v2 requirement it describes is load
> bearing rather than hypothetical: moving off `eleven_flash_v2` would drop
> tagged words entirely.
>
> **Corrected again, Sep 13 2026.** That held for flash v2.5, not every model. On
> eleven_v3 the tag is kept: "Morning, Lethanial." ran 1.04s with it against
> 0.72s for "Morning." alone, and deliberately wrong phonemes ran longer at
> 1.28s. Durations suggest v3 honors the tag; by ear, plain and tagged sound
> about the same, and whether either is right is still open.

### Changing a pronunciation

`scripts/pronounce.py`. Aliases live in the database and `speak()` reads them
per sentence, so a change is live on the next thing Nova says with **no
restart**.

```bash
python3 scripts/pronounce.py list
python3 scripts/pronounce.py try Lethanial Luthanyull Lah-than-yull
python3 scripts/pronounce.py set Lethanial Luthanyull
```

## ElevenLabs quirks

ElevenLabs specific:
- Emma voice catalog deprecates Dec 31 2026, save to "My Voices" before then
- Python's `stdin.write()` to the aplay subprocess buffers up to 64KB by default.
  ALWAYS call `flush()` after every write or you get ~1.5s phantom latency.
- aplay's ALSA buffer holds ~185ms of audio after writing stops (relevant for
  future barge in support)
- v3 stability above 0.7 makes it ignore audio tags
- v3 has no WebSocket support. It accepts `use_speaker_boost` and any stability; whether it honors them is unmeasured (checked Sep 13 2026)
- v3 ignores `speed`: 1.0 and 1.1 rendered to identical lengths
- `optimize_streaming_latency` is deprecated in 2026, do not use

## Related

- Changing `stability` invalidates the rendered phrase bank. The render and trim workflow is above; the voice id and model are declared in [CLAUDE.md](../CLAUDE.md).
- An ack blocks and the chime does not, which is a latency decision as much as a voice one: [LATENCY.md](LATENCY.md)
- Why an ack cannot overlap capture: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#endpointing)
