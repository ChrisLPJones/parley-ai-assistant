"""Kokoro voice names -> grouped, friendly labels for the settings window (pure, no Tk, no models).

A voice is `<language letter><gender letter>_<name>`, e.g. `af_heart` = American English, female, "Heart".
"""
LANGS = {"a": "American English", "b": "British English", "e": "Spanish", "f": "French", "h": "Hindi", "i": "Italian",
         "j": "Japanese", "p": "Brazilian Portuguese", "z": "Mandarin Chinese"}
GENDERS = {"f": "female", "m": "male"}
SAMPLE_TEXT = "Hello, I'm {name}. This is how I sound at this speed."


def friendly_name(voice: str) -> str:
    """'af_heart' -> 'Heart', 'zf_xiaobei' -> 'Xiaobei'."""
    return voice.partition("_")[2].replace("_", " ").title() or voice


def group_label(voice: str) -> str:
    """'af_heart' -> 'American English, female'; a voice with an unknown prefix -> 'Other'."""
    lang, gender = LANGS.get(voice[:1]), GENDERS.get(voice[1:2])
    if lang is None or gender is None or voice[2:3] != "_":
        return "Other"
    return f"{lang}, {gender}"


def group_voices(names) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(group label, [(voice id, friendly name), ...]), ...]: languages in LANGS order, female before male, 'Other' last."""
    order = {k: i for i, k in enumerate(LANGS)}
    groups: dict[str, list[tuple[str, str]]] = {}
    for v in sorted(set(names), key=lambda v: (order.get(v[:1], 99), v[1:2] != "f", v)):
        groups.setdefault(group_label(v), []).append((v, friendly_name(v)))
    other = groups.pop("Other", None)
    out = list(groups.items())
    if other:
        out.append(("Other", other))
    return out


def sample_text(voice: str) -> str:
    return SAMPLE_TEXT.format(name=friendly_name(voice))
