# App handoff: reviewing duplicate memories

> Written Sep 28 2026 for building the screens in the miles-app repo. The
> backend is live in this repo; nothing here needs a backend change. Endpoint
> rules are also in [INFRASTRUCTURE.md](INFRASTRUCTURE.md#fastapi-endpoints),
> and why duplicates are never merged automatically is in
> [BRAIN.md](BRAIN.md#duplicates-are-not-merged-automatically-measured-sep-28-2026).

Up: [CLAUDE.md](../CLAUDE.md)

## What this adds to the app

Nova sometimes stores the same fact twice in different words. Two different
facts about the same person look just as similar to the code, so the app shows
possible pairs and Lethanial decides. Two places:

1. **Possible duplicates**, a section on the Memories screen: pairs of stored
   memories, each with Merge, Keep both, and Keep one.
2. **A hint on pending memories**: when a memory waiting for approval is close
   to a stored one, the stored one is shown under it, with Replace it, Keep
   both, and Discard.

All requests use the same Bearer token as the existing memory endpoints.

## Endpoints

### `GET /memories/similar`

```json
{"pairs": [
  {"similarity": 0.94,
   "a": {"id": 345, "content": "Obsession is one of his top three movies of 2026, ..."},
   "b": {"id": 346, "content": "Obsession, Odyssey, and Spider-Man: Brand New Day are ..."}}
]}
```

Closest first. `a` is always the older memory. Pairs he chose Keep both on are
never returned. An empty list is normal.

### `POST /memories/similar/dismiss`

Body `{"a_id": 21, "b_id": 22}`, order does not matter. Returns
`{"dismissed": true}`. 404 when either is not a current memory.

### `POST /memories/merge/draft`

Body `{"a_id": 345, "b_id": 346}`. Returns
`{"content": "one sentence holding both", "drafted": true}`. **Stores
nothing.** Takes one to three seconds (a Claude call), so show a spinner in the
text box. When `drafted` is false the model was unreachable and `content` is
just the longer memory; still usable as a starting point.

### `POST /memories/merge`

Body `{"keep_id": 345, "drop_id": 346, "content": "final text"}`. Returns
`{"id": 350, "replaced": 345, "deleted": 346}`.

`keep_id` is superseded by the new text (its history keeps the old wording,
visible through the existing `GET /memories/{id}/history`), and `drop_id` is
deleted. One transaction: it either all happens or none of it does. 400 for
empty text, 404 unless both are current, different memories. **The merged
memory has a new id**, so refresh the list rather than editing rows in place.

### `GET /memories/pending`, one new field

Each item now carries `similar_to`:

```json
{"id": 352, "content": "Not pledging D9 this year.", "category": null,
 "source": "implicit", "confidence": "high",
 "similar_to": {"id": 186, "content": "He plans to pledge a D9 fraternity in Fall 2026, ...",
                "similarity": 0.81}}
```

`similar_to` is `null` when nothing stored is close. Decode it as optional.

## Screen 1: Possible duplicates

Load `GET /memories/similar` with the Memories screen. Hide the section when
there are no pairs. One card per pair, both texts in full (they are the whole
point; do not truncate), older on top.

| Button | Does |
|---|---|
| **Merge** | Opens an editor prefilled from `/memories/merge/draft`. Save sends `/memories/merge` with `keep_id` = the older (`a`), `drop_id` = `b`, and the edited text. |
| **Keep both** | `/memories/similar/dismiss`. The card goes away for good. |
| **Keep one** | He taps the one to keep; `DELETE /memories/{id}` on the other. |

After any of them, reload the section and the memory list. The draft editor
needs Cancel, and cancelling must store nothing (the draft endpoint already
stores nothing, so this is only about not calling merge).

Most pairs listed today are different facts about the same family members, so
Keep both will be the most used button. Make it as easy to reach as Merge.

Do not show `similarity` as a verdict, and do not sort or colour by it beyond
the order the server gives. A 0.83 can be a duplicate or two different facts;
the number cannot tell them apart, which is why he is being asked.

## Screen 2: the pending hint

On each pending card with a non null `similar_to`, show the stored memory
underneath, labelled something like "Already stored:". The card's actions become:

| Button | Does |
|---|---|
| **Replace it** | The pending fact updates the stored one. Editor prefilled with the pending text; save sends `/memories/merge` with `keep_id` = `similar_to.id`, `drop_id` = the pending id. Optionally a "Combine instead" button in the editor that fills it from `/memories/merge/draft`. |
| **Keep both** | `/memories/similar/dismiss` with the two ids, **then** the existing `POST /memories/{id}/approve`. Dismiss first, so the approved fact never shows up in Screen 1 beside the one he just judged. |
| **Discard** | The existing `DELETE /memories/{id}`. |

Cards with `similar_to` null keep today's Approve and Discard.

## Done means

- A pair can be merged, kept, or reduced to one, and the list reflects it after reload.
- Keep both on either screen means that pair never appears in Screen 1 again.
- Replace it on a pending fact leaves one active memory whose history shows the old wording.
- Cancelling any editor changes nothing on the server.
- An older app build still works: it ignores `similar_to` and the new routes.
