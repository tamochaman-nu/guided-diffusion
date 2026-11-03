"""
Attribute labels (gender / face view) for attribute-conditioned training.

Labels come from a manifest, by default <data_dir>/metadata.jsonl, one JSON
object per image:

    {"file": "sub/0001.jpg", "gender": "male", "view": "profile",
     "tags": {"1boy": 0.97, "profile": 0.88, ...}}

- "file": path relative to data_dir (a bare file name also matches if unique).
- "gender": "female" | "male" | "unknown"
- "view":   "front" | "three_quarter" | "profile" | "unknown"
- "tags" (optional): tagger output, a {tag: confidence} dict or a list of tags.
  Used only when "gender"/"view" are missing (see attrs_from_tags).

Images without an entry, or with "unknown", get the label -1 for that
attribute, which the model treats like the CFG null label (all-zero vector).
"""

import json
import os

ATTRIBUTES = {
    "gender": ["female", "male"],
    "view": ["front", "three_quarter", "profile"],
}
ATTR_NAMES = list(ATTRIBUTES)
ATTR_DIMS = tuple(len(v) for v in ATTRIBUTES.values())
UNKNOWN = -1

# Conservative fallback from danbooru-style tags (only when the manifest has no
# explicit field). Danbooru tags have no reliable "front" / "three-quarter" tag,
# so only "profile" is derived for the view.
_TAG_THRESHOLD = 0.5
_GENDER_TAGS = {"1girl": "female", "1boy": "male"}
_MULTI_TAGS = {"2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls",
               "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple_boys"}
_PROFILE_TAGS = {"profile"}


def _tag_set(tags):
    if tags is None:
        return set()
    if isinstance(tags, dict):
        return {t for t, c in tags.items() if c is None or float(c) >= _TAG_THRESHOLD}
    return set(tags)


def attrs_from_tags(tags):
    """Map tagger output to {"gender": ..., "view": ...} (unknown when ambiguous)."""
    ts = _tag_set(tags)
    genders = {g for t, g in _GENDER_TAGS.items() if t in ts}
    gender = genders.pop() if len(genders) == 1 and not (ts & _MULTI_TAGS) else "unknown"
    view = "profile" if ts & _PROFILE_TAGS else "unknown"
    return {"gender": gender, "view": view}


def encode(record):
    """Record -> list of label indices (UNKNOWN for missing / unknown)."""
    fallback = None
    out = []
    for name, values in ATTRIBUTES.items():
        v = record.get(name)
        if v is None:
            if fallback is None:
                fallback = attrs_from_tags(record.get("tags"))
            v = fallback[name]
        if v != "unknown" and v not in values:
            raise ValueError(f"invalid {name}={v!r} for {record.get('file')}")
        out.append(values.index(v) if v in values else UNKNOWN)
    return out


def parse_attribute(name, value):
    """CLI value ("male", "profile", "any"/"unknown"/"") -> label index."""
    if value in ("", "any", "unknown", None):
        return UNKNOWN
    return ATTRIBUTES[name].index(value)


def load_manifest(path, image_paths, data_dir):
    """Return a [len(image_paths)] list of label lists for the given images."""
    by_rel, by_base, dup_base = {}, {}, set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            labels = encode(rec)
            rel = os.path.normpath(rec["file"])
            by_rel[rel] = labels
            base = os.path.basename(rel)
            if base in by_base:
                dup_base.add(base)
            by_base[base] = labels
    unknown = [UNKNOWN] * len(ATTRIBUTES)
    out = []
    for p in image_paths:
        rel = os.path.normpath(os.path.relpath(p, data_dir))
        base = os.path.basename(rel)
        if rel in by_rel:
            out.append(by_rel[rel])
        elif base in by_base and base not in dup_base:
            out.append(by_base[base])
        else:
            out.append(unknown)
    return out


def summarize(labels):
    """Counts per attribute value (for logging)."""
    s = {}
    for i, (name, values) in enumerate(ATTRIBUTES.items()):
        counts = {v: 0 for v in values + ["unknown"]}
        for lab in labels:
            counts[values[lab[i]] if lab[i] >= 0 else "unknown"] += 1
        s[name] = counts
    return s
