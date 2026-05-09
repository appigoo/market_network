# modules/telegram_module.py
# ============================================================
# Telegram 通知模組
# ============================================================

import os
import requests
import streamlit as st
import hashlib
import time


def _get_telegram_config():
    token   = os.environ.get("TELEGRAM_TOKEN", st.session_state.get("telegram_token", ""))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", st.session_state.get("telegram_chat_id", ""))
    return token, chat_id


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    """發送 Telegram 通知"""
    token, chat_id = _get_telegram_config()
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": parse_mode,
        }, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _alert_hash(message: str) -> str:
    return hashlib.md5(message.encode()).hexdigest()[:12]


def send_deduped_alert(message: str, cooldown_minutes: int = 30) -> bool:
    """去重發送：同一警報在冷卻期內只發一次"""
    key = f"tg_sent_{_alert_hash(message)}"
    last_sent = st.session_state.get(key, 0)
    now = time.time()
    if now - last_sent < cooldown_minutes * 60:
        return False  # 冷卻中
    ok = send_telegram(message)
    if ok:
        st.session_state[key] = now
    return ok
