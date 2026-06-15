# =========================================================================
#  Be More Agent · 提示词
#  从 agent.py 抽出。档2: 每状态窄 prompt + few-shot 范例后续在此扩展。
# =========================================================================

from config import CURRENT_CONFIG

# --- SYSTEM PROMPT ---
# 档1: 去掉原英文工具调用 prompt（拍照/搜索已删）。这是 config.json 缺失时的兜底；
# 正式的睡前梳理 prompt（窄 prompt + few-shot）在档2扩展。
BASE_SYSTEM_PROMPT = """你是一个睡前陪伴机器人，帮用户在睡前梳理情绪。
说话温和、简短，每次回应不超过两句话。
只负责接住用户当下这一句，不出主意、不深挖、不在睡前帮用户解决烦心事。"""

SYSTEM_PROMPT = CURRENT_CONFIG.get("system_prompt", BASE_SYSTEM_PROMPT) + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")
