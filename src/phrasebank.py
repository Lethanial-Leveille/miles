"""Victoria, rendered once while online and played from disk with no network.

Everything before the API call in a turn is already local: wake word, VAD,
Whisper, Resemblyzer. Only two steps need the network, and they fail in
different ways. Claude raises, which voice_main now catches. ElevenLabs fails
silently in tts.speak, which is why an offline Nova used to give no sound at
all rather than an explanation.

This closes that second hole for the fixed set of things Nova says that never
needed a language model to produce. A local TTS engine could also speak them,
but it would be a different voice wearing her name. These files are actually
her, because they are her output saved rather than re-fetched.

PHRASES is the source of truth and is versioned. The audio is derived from it
by scripts/render_phrases.py and lives under data/, which is gitignored, so a
fresh clone renders its own with its own key.

Playback is deliberately the same path play_chime uses: aplay against a WAV.
ElevenLabs returns pcm_22050 and the chime is 22050Hz mono 16 bit, so the two
are already the same format and nothing here resamples or converts.
"""

import os
import random
import subprocess
import time
from datetime import datetime

import timing
import tts
from config import PHRASE_DIR
from parsing import words_for_number

# Wake acknowledgements are mixed with the chime rather than replacing it,
# because they are not free the way the chime is. The chime is a tone, and
# webrtcvad at mode 2 does not read a tone as speech, so it overlaps recording
# harmlessly. These are speech: overlapped, they trip VAD onset, land at the
# head of the transcript, and end up in the clip Resemblyzer scores against his
# voiceprint. So they play to completion before the mic opens, and their length
# sits between the wake word and the command.
#
# Measured after trimming, against the chime's 0.320s: "Yep?" 0.358s, "Yeah?"
# 0.398s, up to "Right here." at 0.833s. So the short ones are within 40ms of
# the chime and genuinely free; the long ones cost half a second and are here
# for variety rather than for every wake. Re-measure with
# `render_phrases.py list` after editing this dict, because the ordering above
# is by length and a new phrase will not respect it.
PHRASES = {
    'ack': [
        "Yeah?",
        "Mhm?",
        "Yes?",
        "Yep?",
        "Go ahead.",
        "Right here.",
        "What's up?",
        "I'm listening.",
    ],

    # One exception, eight causes. netcheck.diagnose picks between these, and
    # every set here is worth having rather than one vague apology: "I've lost
    # wifi" is a false statement when the truth is that Anthropic is down, and
    # being wrong about its own state is worse than saying less. Each one names
    # only what has actually been observed, never what was left over.
    #
    # Precise rather than plain on purpose, all the way down. He is the only
    # person who hears these, he is a computer engineer, and "something is
    # wrong" would send him to power cycle a router that is working fine.
    #
    # Nothing follows any of these, so each has to close the exchange on its own
    # rather than leave him waiting for more.
    #
    # netcheck.CAUSES is the list these have to cover, and a test enforces it.
    'no_wifi': [
        "I've lost the wifi connection.",
        "I'm not on the network right now, so that will have to wait.",
        "The wifi is down. Try me again once it's back.",
    ],
    # The kernel never put a packet on the wire. Named separately from a
    # timeout because they point at different halves of the house: no route is
    # this machine's routing table, a timeout is everything past the door.
    'no_route': [
        "I'm on the wifi, but there's no route out of here.",
        "The link is up and there's nowhere for it to send anything. No route.",
    ],
    'net_timeout': [
        "I'm on the network, but nothing is coming back. It's all timing out.",
        "Packets are going out and nothing is answering. That's a timeout.",
    ],
    # Kept for the failures that are real and did not name themselves. Its
    # wording was already the general case, so it needed no rewrite when the
    # specific causes moved out above it.
    'no_internet': [
        "I'm on the wifi, but nothing is getting out to the internet.",
        "The network is up and the connection isn't going anywhere.",
    ],
    'no_dns': [
        "The network is up, but DNS isn't resolving.",
        "I can route out, but nothing is resolving. That's DNS.",
    ],
    # A refusal is the most informative failure of the set: something was
    # listening and said no. Worth its own sentence, because it is the one that
    # means the network is entirely fine.
    'api_refused': [
        "I can reach Anthropic, and it's refusing the connection.",
        "The route and the name both work. Anthropic is refusing me.",
    ],
    'api_timeout': [
        "I can resolve Anthropic, but the connection just times out.",
        "Anthropic has an address and nothing behind it is answering.",
    ],
    'api_down': [
        "My connection is fine. It's the model I can't reach.",
        "Everything on my end is up, so the API is what's unreachable.",
    ],

    # Reached the API and it failed anyway, which usually means the network is
    # up and tts.speak would work. Rendered regardless, because the one time it
    # matters is the time that assumption is wrong.
    'error': [
        "Something went wrong on my end. Try me again in a moment.",
        "That didn't go through. Give it another shot.",
    ],

    # Spoken when local intent ends the conversation, in place of the goodbye
    # Claude would have written. Varied for the same reason the dismiss tool
    # description asks Claude for variety: one stock farewell every time is the
    # tell that nobody is home.
    # Index 0 is night only, see NIGHT_ONLY. The rest carry any hour, which is
    # why there are five of them: outside night they are the whole set.
    # Said the moment a slow tool starts, on a voice turn where Nova wrote no
    # lead in of her own; see brain._BRIDGES. Tool turns were half of all turns
    # on Sep 16 2026, with a median 5258ms before any sound and 8022ms to finish.
    # Short, because the answer waits for the line to end, and committing to
    # nothing, because the result is not known yet.
    'bridge_calendar': [
        "Checking your calendar.",
        "Let me look at your calendar.",
        "One sec, pulling up your calendar.",
    ],
    'bridge_health': [
        "Checking your Oura numbers.",
        "Let me pull up your ring data.",
        "One sec, looking at your Oura.",
    ],
    'bridge_weather': [
        "Checking the weather.",
        "Let me check the forecast.",
        "One sec, looking at the weather.",
    ],

    'dismiss': [
        "Goodnight, Lethanial.",
        "Anytime.",
        "Talk soon.",
        "Sure thing. I'm here if you need me.",
        "Alright. Catch you later.",
        "Any time at all.",
    ],
}


# Timer values worth enumerating. Anything outside this falls through to
# Claude, which is a correct outcome rather than a gap: a fall through costs
# latency and nothing else.
TIMER_VALUES = ([(n, 'minutes') for n in range(1, 61)]
                + [(n, 'seconds') for n in (10, 15, 20, 30, 45)]
                + [(n, 'hours') for n in (1, 2, 3)])


def timer_key(amount, unit):
    return f"timer_set_{amount}_{unit}"


def timer_text(amount, unit):
    """One complete sentence per value, which is the whole point.

    The obvious way to build this is a carrier phrase plus a number clip plus a
    unit clip, played back to back. That is the trap. A word rendered alone
    gets terminal prosody: "ten" spoken by itself falls in pitch like the end
    of a sentence, because in isolation it is one. Three such clips glued
    together are unmistakably a phone menu, in Victoria's timbre but not her
    delivery, and there is no coarticulation across the joins either.

    Enumerating whole sentences keeps every clip a genuine single utterance.
    The space is small enough that this costs about 6MB, and a space small
    enough to enumerate is the same condition that makes an intent worth
    handling locally at all."""
    singular = unit[:-1] if amount == 1 else unit
    return f"Timer set for {words_for_number(amount)} {singular}."


PHRASES.update({timer_key(a, u): [timer_text(a, u)] for a, u in TIMER_VALUES})


# The clock, to five minutes. Twelve hours by twelve slots is 144 clips, which
# is small enough to enumerate whole sentences and so keeps real prosody. Exact
# minutes would be 720, and "about" on every reading grates, so the reading is
# rounded to the nearest five and stated plainly. Two minutes is inside what
# people round to when saying a time out loud anyway.
TIME_VALUES = [(h, m) for h in range(1, 13) for m in range(0, 60, 5)]


def time_key(hour, minute):
    return f"time_{hour}_{minute}"


def time_text(hour, minute):
    if minute == 0:
        spoken = "o'clock"
    elif minute < 10:
        # "four oh five", not "four five".
        spoken = f"oh {words_for_number(minute)}"
    else:
        spoken = words_for_number(minute)
    return f"It's {words_for_number(hour)} {spoken}."


def time_key_for(now):
    """Round to the nearest five minutes, rolling the hour when it lands on 60."""
    minute = int(round(now.minute / 5.0) * 5)
    hour   = now.hour
    if minute == 60:
        minute = 0
        hour += 1
    hour = hour % 12 or 12
    return time_key(hour, minute)


PHRASES.update({time_key(h, m): [time_text(h, m)] for h, m in TIME_VALUES})


# Cancelling is only handled locally when it is unambiguous, so the confirmation
# never has to name which reminder went.
PHRASES['cancelled'] = [
    "That's cancelled.",
    "Done, it's cancelled.",
    "Cancelled.",
]


def path_for(key, index):
    return os.path.join(PHRASE_DIR, f"{key}.{index}.wav")


# Nine at night to five in the morning.
NIGHT_HOURS = set(range(21, 24)) | set(range(0, 5))

# Variants that only fit part of the day, by index within their key. A
# conversation ends at any hour, and "goodnight" at two in the afternoon is
# worse than no farewell at all. Kept as an index filter rather than a separate
# key so a variant already auditioned and committed keeps its file.
NIGHT_ONLY = {'dismiss': {0}}        # "Goodnight, Lethanial."

# Replies that answer gratitude rather than close a conversation. "Never mind"
# is a retraction, not a thank you, and "Any time at all" in reply to it reads
# as not having listened. Same index filter as NIGHT_ONLY, for the same reason:
# these files are already rendered and must keep their numbering.
THANKS_ONLY = {'dismiss': {1, 5}}    # "Anytime.", "Any time at all."


def rendered(key, hour=None, thanked=False):
    """(index, path) for every variant of `key` playable in this context.

    Partial renders are normal and fine: a phrase added to PHRASES is playable
    as soon as any one of its variants exists, and the rest fill in the next
    time the render script runs."""
    hour = datetime.now().hour if hour is None else hour
    blocked = set()
    if hour not in NIGHT_HOURS:
        blocked |= NIGHT_ONLY.get(key, frozenset())
    if not thanked:
        blocked |= THANKS_ONLY.get(key, frozenset())

    found = []
    for index in range(len(PHRASES.get(key, ()))):
        if index in blocked:
            continue
        path = path_for(key, index)
        if os.path.exists(path):
            found.append((index, path))
    return found


def play(key, hour=None, thanked=False, bridge=False):
    """Play one eligible variant at random.

    Returns the text of what was played, or None if nothing is rendered. The
    text is returned rather than a bool because the caller writes it to history,
    and returning a bool meant the caller had to guess which variant ran. It
    guessed index zero, so history could record "Goodnight, Lethanial." on a
    turn where she actually said "Talk soon."

    Always blocking, and always under speak_lock. Every caller either speaks
    before the mic opens, where Nova's own voice in the capture buffer corrupts
    both the transcript and the verification score, or is the entire response to
    a turn. Neither wants a timer alert talking over it."""
    started = time.monotonic()
    choices = rendered(key, hour, thanked)
    if not choices:
        return None

    index, path = random.choice(choices)
    with tts.speak_lock:
        # Inside the lock, not before it. Waiting on speak_lock is time the room
        # spends in silence exactly like every other stage, and measuring above
        # it would hide a contended turn behind a number that looks instant.
        #
        # Ignored unless a turn is open, so the wake ack and the offline
        # apologies cost nothing here: the ack plays before begin_turn, and
        # _say_cached runs after abandon_turn.
        if bridge:
            timing.note_bridge()
        else:
            timing.note_local_audio((time.monotonic() - started) * 1000.0)
        subprocess.run(
            ["aplay", "-D", tts.SPEAKER_DEVICE, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return PHRASES[key][index]
