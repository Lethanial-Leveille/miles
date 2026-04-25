import anthropic
from config import DB_PATH
from prompts import build_enhanced_prompt, SYSTEM_PROMPT
from database import save_message, get_memories, get_recent_messages, save_memory
from parsing import extract_memories, extract_actions
from actions import execute_actions

claude = anthropic.Anthropic()


def ask_nova(user_text, device="pi"):
    save_message("user", user_text, device=device)

    memory_rows     = get_memories()
    enhanced_prompt = build_enhanced_prompt(memory_rows)
    recent          = get_recent_messages(20)

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=enhanced_prompt,
        messages=recent,
    )
    nova_text = response.content[0].text

    clean, explicit_mems, implicit_mems = extract_memories(nova_text)
    for mem in explicit_mems:
        save_memory(mem, source="explicit")
    for mem in implicit_mems:
        save_memory(mem, source="implicit")

    clean, action_list = extract_actions(clean)

    if action_list:
        results      = execute_actions(action_list)
        needs_data   = any(r["type"] == "weather" for r in results)

        if needs_data:
            data_block = "\n".join(
                f"{r['type'].upper()} DATA: {r['data']}" for r in results
            )
            followup_messages = recent + [
                {"role": "assistant", "content": clean},
                {"role": "user", "content": (
                    f"[SYSTEM: Here is the data you requested]\n{data_block}\n"
                    "Deliver this information naturally as Nova. Stay in character. Keep it concise."
                )},
            ]
            followup = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=followup_messages,
            )
            clean = followup.content[0].text
            clean, _, _ = extract_memories(clean)
            clean, _    = extract_actions(clean)

    save_message("assistant", clean, device=device)
    return clean
