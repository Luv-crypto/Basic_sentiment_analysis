# Basic_sentiment_analysis
Sentiment analysis on 150 different articles
from __future__ import annotations

import os
import sys
import re
import json
import time
import base64
import hashlib
import secrets
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# -------- generic RAG core ----------
# make sure this file exists in your project
# and exposes:
#   ingest_documents(path, user_id=..., ...)
#   smart_query(question, user_id=..., return_media=True)
from generic_rag_core import ingest_documents, smart_query

load_dotenv()

# ============================================================
# PATHS / CONFIG
# ============================================================
ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "users.db"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOGO = ROOT / "assets" / "logo.png"   # optional
AGENT_SCRIPT = ROOT / "agent.py"              # keep your existing agent script here

st.set_page_config(page_title="AI Workspace", layout="wide")


# ============================================================
# OPTIONAL HEADER / LOGO
# ============================================================
def get_base64_img(img_path: str | Path) -> str:
    img_path = Path(img_path)
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_header():
    logo_b64 = get_base64_img(DEFAULT_LOGO)
    if logo_b64:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                <img src="data:image/png;base64,{logo_b64}" width="52">
                <div>
                    <div style="font-size:28px;font-weight:700;">AI Workspace</div>
                    <div style="color:#666;font-size:14px;">Agentic MDAO + Deep Research</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("AI Workspace")


# ============================================================
# DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            pw_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mode TEXT,
            role TEXT,
            payload TEXT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, pw_hash) VALUES (?, ?)",
            (username, hash_pw(password))
        )
        conn.commit()
        return True, "Registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def login_user(username: str, password: str) -> tuple[bool, int | None, str]:
    username = username.strip()
    conn = get_db()
    row = conn.execute(
        "SELECT id, pw_hash FROM users WHERE username=?",
        (username,)
    ).fetchone()
    conn.close()

    if not row or row[1] != hash_pw(password):
        return False, None, "Bad credentials."

    return True, int(row[0]), "Login successful."


def save_chat(user_id: int, mode: str, role: str, payload):
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False)

    conn = get_db()
    conn.execute(
        "INSERT INTO chats (user_id, mode, role, payload) VALUES (?, ?, ?, ?)",
        (user_id, mode, role, payload)
    )
    conn.commit()
    conn.close()


def load_chats(user_id: int, mode: str) -> list[tuple[str, str]]:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, payload FROM chats WHERE user_id=? AND mode=? ORDER BY id",
        (user_id, mode)
    ).fetchall()
    conn.close()
    return rows


def clear_chats(user_id: int, mode: str):
    conn = get_db()
    conn.execute("DELETE FROM chats WHERE user_id=? AND mode=?", (user_id, mode))
    conn.commit()
    conn.close()


# ============================================================
# SESSION STATE
# ============================================================
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

if "uid" not in st.session_state:
    st.session_state.uid = None

if "username" not in st.session_state:
    st.session_state.username = None


# ============================================================
# COMMON HELPERS
# ============================================================
MEDIA_TOKEN_RE = re.compile(r'^\s*<<(img|tbl):[^>]+>>\s*$', re.MULTILINE)


def clean_answer_text(text: str) -> str:
    return MEDIA_TOKEN_RE.sub("", text).strip()


def render_media(media: List[Tuple[str, str]] | List[list]):
    for item in media:
        kind, path = item
        p = Path(path)
        if not p.exists():
            continue

        if kind == "img":
            st.image(str(p), use_container_width=True)
        elif kind == "tbl":
            with st.expander(f"Table: {p.name}", expanded=False):
                st.markdown(p.read_text(encoding="utf-8"))


def render_saved_message(role: str, payload: str):
    if role == "user":
        st.markdown(payload)
        return

    try:
        data = json.loads(payload)
        answer = data.get("answer", "")
        media = data.get("media", [])
        st.markdown(answer)
        render_media(media)
    except Exception:
        st.markdown(payload)


# ============================================================
# AUTH UI FOR DEEP RESEARCH
# ============================================================
def deep_research_auth():
    if st.session_state.uid is not None:
        return

    st.subheader("Deep Research Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="dr_login_user")
        p = st.text_input("Password", type="password", key="dr_login_pw")
        if st.button("Login", key="dr_login_btn", use_container_width=True):
            ok, uid, msg = login_user(u, p)
            if ok:
                st.session_state.uid = uid
                st.session_state.username = u.strip()
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        u = st.text_input("New Username", key="dr_reg_user")
        p = st.text_input("New Password", type="password", key="dr_reg_pw")
        if st.button("Register", key="dr_reg_btn", use_container_width=True):
            ok, msg = register_user(u, p)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.stop()


# ============================================================
# AGENTIC MDAO TAB
# keeps your old style: subprocess + streamed output
# ============================================================
def run_agentic_mode():
    st.subheader("Agentic MDAO")

    # replay history
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Enter instructions for Agentic MDAO...", key="agent_input")
    if not prompt:
        return

    st.session_state.agent_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_output = ""

        try:
            process = subprocess.Popen(
                [sys.executable, str(AGENT_SCRIPT), prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                full_output += line
                response_placeholder.markdown(full_output + "▌")

            process.wait()
            full_output = full_output.strip() or "No response returned."
            response_placeholder.markdown(full_output)

        except Exception as e:
            full_output = f"Error running agent script: {e}"
            response_placeholder.markdown(full_output)

    st.session_state.agent_history.append({"role": "assistant", "content": full_output})


# ============================================================
# DEEP RESEARCH TAB
# adapted from your Flask flow, but compact for Streamlit
# ============================================================
def run_deep_research_mode():
    deep_research_auth()

    uid = st.session_state.uid
    mode = "deep_research"

    user_dir = UPLOAD_DIR / f"user_{uid}"
    user_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Deep Research")

    # -------- sidebar controls --------
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username}")

        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="deep_pdf_upload"
        )

        if uploaded_pdf and st.button("Save + Ingest", use_container_width=True, key="deep_ingest_btn"):
            save_path = user_dir / f"{secrets.token_hex(6)}_{uploaded_pdf.name}"
            save_path.write_bytes(uploaded_pdf.getbuffer())

            with st.spinner(f"Ingesting {uploaded_pdf.name}..."):
                ingest_documents(str(save_path), user_id=uid)

            st.success(f"Indexed: {uploaded_pdf.name}")
            st.rerun()

        user_pdfs = sorted(user_dir.glob("*.pdf"))
        if user_pdfs:
            st.markdown("**Uploaded PDFs**")
            for f in user_pdfs:
                st.caption(f.name)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Chat", use_container_width=True, key="deep_clear_chat"):
                clear_chats(uid, mode)
                st.rerun()

        with col2:
            if st.button("Logout", use_container_width=True, key="deep_logout"):
                st.session_state.uid = None
                st.session_state.username = None
                st.rerun()

    # -------- replay saved history --------
    for role, payload in load_chats(uid, mode):
        with st.chat_message("assistant" if role == "assistant" else "user"):
            render_saved_message(role, payload)

    # -------- ask question --------
    prompt = st.chat_input("Ask about your uploaded PDFs...", key="deep_chat_input")
    if not prompt:
        return

    save_chat(uid, mode, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, media = smart_query(prompt, user_id=uid, return_media=True)

            clean_answer = clean_answer_text(answer)
            st.markdown(clean_answer)
            render_media(media)

            save_chat(
                uid,
                mode,
                "assistant",
                {"answer": clean_answer, "media": media}
            )

        except Exception as e:
            err = f"Server error: {e}"
            st.error(err)
            save_chat(uid, mode, "assistant", {"answer": err, "media": []})


# ============================================================
# MAIN APP
# ============================================================
render_header()

app_mode = st.sidebar.radio("Choose Mode", ["Agentic MDAO", "Deep Research"])

if app_mode == "Agentic MDAO":
    run_agentic_mode()
else:
    run_deep_research_mode()
