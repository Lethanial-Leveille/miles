"""What verify_voice decides, with nothing else: no logging, no printing, no
voiceprint learning, no timing.

Split out so the speculative turn can ask the same question of the early
embedding before the endpoint, and must get the answer verify_voice will give
after it. Two copies of this rule would drift, and the day they disagreed a
speculation would send a personal prompt for a voice the real check refuses.

Kept out of audio.py so it can be imported and tested while miles-voice holds
the microphone, the same reason early_verify.py exists.
"""

from typing import NamedTuple, Optional

import numpy as np

import config

# verify_voice outcomes. NO_AUDIO is distinct from REJECTED on purpose: one
# means "not your voice", the other means "there was no voice to compare".
VERIFIED = 'verified'
REJECTED = 'rejected'
NO_AUDIO = 'no_audio'
RETRY = 'retry'

# Resemblyzer computes one partial utterance per 1.6s window and zero pads the
# last one. Below roughly a third of that there is not enough voiced audio for
# the embedding to mean anything.
MIN_EMBED_SECONDS = 0.5

# Below this, an embedding is too unstable to reject anyone on. Measured: a
# 0.9s follow up of "Okay, thank you" scored 0.460 against a voiceprint whose
# median live score is 0.781, and was refused. The utterance was genuine; the
# embedding simply had too little to work with.
#
# Short conversational follow ups ("yeah", "thanks", "what about tomorrow")
# are exactly the turns that cannot be scored reliably, so scoring them at all
# produces false rejections and nothing else.
MIN_TRUSTWORTHY_SECONDS = 2.0


class Verdict(NamedTuple):
    outcome: str
    similarity: Optional[float]
    # What verification_log records as the outcome, or None when nothing is
    # logged: an encoder that declined the clip has no score to write down.
    logged_as: Optional[str]


def before_embedding(embedded_seconds, session_trusted=False):
    """The verdict that needs no embedding, or None when the clip must be scored.

    Guarded before embedding, not after: embed_utterance() zero pads whatever it
    is handed up to one partial and returns a confident looking vector for
    silence, so there is no downstream signal that the input was empty.

    A clip too short to score, inside a session that already authenticated on a
    longer one, trusts the session rather than reject a genuine speaker on an
    embedding that cannot support the decision either way."""
    if embedded_seconds < MIN_EMBED_SECONDS:
        return Verdict(NO_AUDIO, None, NO_AUDIO)
    if session_trusted and embedded_seconds < MIN_TRUSTWORTHY_SECONDS:
        return Verdict(VERIFIED, None, 'session_trust')
    return None


def on_embedding(embedding, voiceprint):
    """Score an embedding against the voiceprint.

    None means the encoder declined the clip as too short. The guard in
    before_embedding normally catches that; reaching here means the two disagree
    by a hair, and inventing a score would be worse than saying there was
    nothing to score.

    An ambiguous band rather than a hard line: scores between the two thresholds
    are the model saying it does not know, and answering that with silence is
    the worst of the three available responses."""
    if embedding is None:
        return Verdict(NO_AUDIO, None, None)
    similarity = float(np.dot(embedding, voiceprint) / (
        np.linalg.norm(embedding) * np.linalg.norm(voiceprint)))
    if similarity >= config.VERIFY_THRESHOLD:
        return Verdict(VERIFIED, similarity, 'scored')
    if similarity >= config.VERIFY_RETRY_THRESHOLD:
        return Verdict(RETRY, similarity, 'retry')
    return Verdict(REJECTED, similarity, 'scored')


def decide(embedded_seconds, embedding, voiceprint, session_trusted=False):
    """The whole decision, for a caller that already has the embedding."""
    return (before_embedding(embedded_seconds, session_trusted)
            or on_embedding(embedding, voiceprint))
