"""One speaker encoder behind one interface, chosen by config.

## Why this exists

Resemblyzer cannot separate Lethanial from his sister. Measured Aug 13 2026 by
scripts/encoder_bench.py against 23 clips of her labelled by ear:

    | | resemblyzer | ecapa |
    | genuine median | 0.764 | 0.542 |
    | impostor max   | 0.843 | 0.179 |
    | EER            | 22.3% |  4.8% |

Her best clip scored above half of his. That is a resolution problem and no
threshold fixes it; same family voices are where a 2019 GE2E model is weakest.

## The property this module exists to enforce

**Embeddings from two encoders are never comparable and must never be mixed.**
Resemblyzer is 256 dimensions and ECAPA is 192, and even if they agreed on
dimension the spaces would be unrelated. Averaging across encoders is the same
class of mistake that poisoned the April voiceprint, except silent: a centroid
built from both would produce plausible looking cosines that mean nothing.

So provenance is written down rather than assumed. A voiceprint is saved with
the name of the encoder that made it, and loading one under a different encoder
is a hard failure at boot instead of a wrong answer at every turn.

## The scales are different and are NOT interchangeable

VERIFY_THRESHOLD = 0.5 sits comfortably below Resemblyzer's 0.764 genuine
median and near ECAPA's 0.542, so carrying it across would reject roughly half
his turns. The threshold has to be re-derived per encoder, never migrated.
config.VERIFY_THRESHOLDS keeps them separate for exactly that reason.
"""

import os

import numpy as np

SR = 16000

# Minimum audio an encoder is allowed to see. Below this the vector is
# dominated by padding and means nothing, and both backends return a confident
# looking result for it rather than an error.
MIN_SECONDS = 0.2

# Declared rather than discovered, so a voiceprint of the wrong size is caught
# by name at load instead of by a shape error somewhere downstream.
DIMENSIONS = {"resemblyzer": 256, "ecapa": 192}


def trim(wav, source_sr=SR):
    """Voice activity trim and level normalize, before any encoder sees it.

    Separate from embedding on purpose. verify_voice has to know how much
    voiced audio survived trimming *before* it decides whether to embed at all:
    that number drives the no audio guard, the session trust cutoff, and
    whether the sample is good enough to learn from. Folding the trim inside
    embed would leave the caller measuring untrimmed audio and applying
    thresholds calibrated on trimmed audio.

    Resemblyzer's preprocess_wav is used whichever encoder is configured. The
    question it answers, how much of this clip is speech, is a property of the
    recording rather than of the model that will embed it, and the duration
    thresholds in audio.py were calibrated against exactly this function."""
    from resemblyzer import preprocess_wav
    return preprocess_wav(wav, source_sr=source_sr)


def _load_resemblyzer():
    from resemblyzer import VoiceEncoder
    encoder = VoiceEncoder()

    def embed(wav):
        """`wav` is already trimmed. See trim() for why that is the caller's
        job rather than this function's."""
        if len(wav) < SR * MIN_SECONDS:
            return None
        vector = encoder.embed_utterance(wav)
        return vector / np.linalg.norm(vector)
    return embed


def _load_ecapa():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain-ecapa"),
        run_opts={"device": "cpu"})

    def embed(wav):
        if len(wav) < SR * MIN_SECONDS:
            return None
        with torch.no_grad():
            vector = model.encode_batch(
                torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
                .unsqueeze(0)).squeeze().numpy()
        return vector / np.linalg.norm(vector)
    return embed


# Same shape as scripts/encoder_bench.py, deliberately: the bench decides which
# encoder to adopt and this runs the one adopted, so they must agree on what
# embedding means or the benchmark measures something production does not do.
# The bench keeps its own copies because it loads two encoders at once to
# compare them, and because it trims inside embed where this separates the two.
LOADERS = {"resemblyzer": _load_resemblyzer, "ecapa": _load_ecapa}

_cache = {}


def get_encoder(name):
    """The embed function for one encoder, loaded once per process.

    Every backend returns a unit length vector at SR, or None when handed less
    than MIN_SECONDS, so callers compare like with like and a dot product is
    already the cosine."""
    if name not in LOADERS:
        raise ValueError(
            f"Unknown speaker encoder {name!r}. Known: {sorted(LOADERS)}")
    if name not in _cache:
        _cache[name] = LOADERS[name]()
    return _cache[name]


# ── voiceprint provenance ──

def _meta_path(voiceprint_path):
    """Sidecar beside the .npy rather than a new container format.

    scripts/voiceprint.py and enroll.py both already read and write the .npy
    directly, and changing that format to carry one string would mean touching
    every reader for something none of them need to parse."""
    return f"{os.path.splitext(voiceprint_path)[0]}.json"


def save_voiceprint(centroid, encoder_name, path):
    """Write the centroid and record which encoder produced it."""
    import json
    from datetime import datetime

    centroid = np.asarray(centroid)
    expected = DIMENSIONS.get(encoder_name)
    if expected is not None and centroid.shape[-1] != expected:
        raise ValueError(
            f"{encoder_name} produces {expected} dimensions but this centroid "
            f"has {centroid.shape[-1]}. Refusing to save a mislabelled "
            f"voiceprint, because the label is the only thing stopping it from "
            f"being scored by the wrong encoder later.")

    np.save(path, centroid)
    with open(_meta_path(path), "w") as f:
        json.dump({"encoder": encoder_name,
                   "dimensions": int(centroid.shape[-1]),
                   "created_at": datetime.now().isoformat()}, f, indent=2)
    return path


def voiceprint_encoder(path):
    """Which encoder built the voiceprint at `path`, or None if unrecorded.

    None means it predates provenance being written down, which is true of
    every voiceprint built before Sep 6 2026. The caller decides what to do
    with that; this does not guess."""
    import json
    try:
        with open(_meta_path(path)) as f:
            return json.load(f).get("encoder")
    except (OSError, ValueError):
        return None


def load_voiceprint(path, encoder_name):
    """Load a voiceprint, refusing one built by a different encoder.

    A hard failure at boot rather than a wrong answer on every turn. Two things
    are checked because either alone can pass while the other fails: the
    recorded name catches two encoders that happen to share a dimension, and
    the dimension catches a voiceprint whose sidecar is missing or stale.

    An unlabelled voiceprint is accepted with a warning rather than refused,
    because the one currently in production has no sidecar and refusing it
    would mean this change cannot be deployed before a re enrollment."""
    centroid = np.load(path)
    recorded = voiceprint_encoder(path)

    if recorded is not None and recorded != encoder_name:
        raise SystemExit(
            f"Voiceprint at {path} was built with {recorded!r} but the "
            f"configured encoder is {encoder_name!r}. Their embedding spaces "
            f"are unrelated, so every similarity score would be meaningless "
            f"rather than merely wrong. Re enroll, or set SPEAKER_ENCODER back "
            f"to {recorded!r}.")

    expected = DIMENSIONS.get(encoder_name)
    if expected is not None and centroid.shape[-1] != expected:
        raise SystemExit(
            f"Voiceprint at {path} has {centroid.shape[-1]} dimensions and "
            f"{encoder_name!r} produces {expected}. This voiceprint was built "
            f"by a different encoder. Re enroll before starting.")

    if recorded is None:
        print(f"NOTE: {path} has no recorded encoder, assuming "
              f"{encoder_name!r}. Re enrolling will record it.", flush=True)

    return centroid
