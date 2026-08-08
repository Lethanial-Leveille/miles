import re

_LEADING_BRACKET = re.compile(r'^\s*\[([^\]]*)\]\s*')
_RECOGNIZED_TAG   = re.compile(r'^(ACTION|MEMORY-EXPLICIT|MEMORY):', re.IGNORECASE)


def strip_leading_bracket_cue(text):
    """Strip a leading bracketed cue Nova should not be emitting, e.g. "[clearly]".
    Leaves ACTION/MEMORY tags alone (defensive: they should already be gone
    by the time this runs, but this must never eat a real tag). Logs when
    it actually strips something, so leakage frequency is visible."""
    match = _LEADING_BRACKET.match(text)
    if not match:
        return text
    inner = match.group(1).strip()
    if _RECOGNIZED_TAG.match(inner):
        return text
    print(f"Bracket leakage stripped: [{inner}]", flush=True)
    return text[match.end():]


def extract_memories(response_text):
    explicit_pattern = r'\[MEMORY-EXPLICIT:\s*(.+?)\]'
    implicit_pattern = r'\[MEMORY:\s*(.+?)\]'

    explicit_memories = re.findall(explicit_pattern, response_text)
    implicit_memories = re.findall(implicit_pattern, response_text)

    clean = re.sub(explicit_pattern, '', response_text)
    clean = re.sub(implicit_pattern, '', clean)
    clean = re.sub(r'  +', ' ', clean).strip()

    return clean, explicit_memories, implicit_memories


def extract_actions(response_text):
    param_pattern  = r'\[ACTION:\s*(\w+)\s*\|\s*(.+?)\]'
    simple_pattern = r'\[ACTION:\s*(\w+)\s*\]'

    actions = re.findall(param_pattern, response_text)
    clean   = re.sub(param_pattern, '', response_text)

    simple_actions = re.findall(simple_pattern, clean)
    clean = re.sub(simple_pattern, '', clean)
    clean = re.sub(r'  +', ' ', clean).strip()

    parsed = []
    for action_type, params_str in actions:
        params = {}
        for param in params_str.split(','):
            if ':' in param:
                key, value = param.split(':', 1)
                params[key.strip()] = value.strip()
            else:
                params['value'] = param.strip()
        parsed.append({"type": action_type.lower(), "params": params})

    for action_type in simple_actions:
        parsed.append({"type": action_type.lower(), "params": {}})

    return clean, parsed
