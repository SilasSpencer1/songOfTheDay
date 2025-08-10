from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from song_of_the_day.config import load_settings
from model.candidate_gen import generate_candidate_pool
from model.rerank import rerank_candidates
from runner.pick_song_of_day import pick_song_of_day
from model.mood import mood_to_feature_targets


def _get_sp_client(use_data_client: bool):
    settings = load_settings()
    if use_data_client:
        from data.spotify_client import SpotifyDataClient

        client = SpotifyDataClient.create()
        user_id = client.user_id
        sp = getattr(client, "sp", None)
        return client, user_id, sp
    else:
        from song_of_the_day.spotify_client import SpotifyService

        svc = SpotifyService(settings)
        sp = getattr(svc, "_sp", None)
        me = sp.current_user() if sp else {"id": "unknown"}
        user_id = me.get("id", "unknown")
        return svc, user_id, sp


def _fetch_recent_and_features(client):
    # Helper to get recent tracks and their audio features
    try:
        recent_df = client.get_recent_tracks(limit=200, days_back=30)
    except TypeError:
        recent_df = client.get_recent_tracks(limit=50)
    if recent_df is None:
        recent_df = pd.DataFrame()
    feats_df = pd.DataFrame()
    if not recent_df.empty:
        feats_df = client.get_audio_features(recent_df["track_id"].dropna().tolist())
    return recent_df, feats_df


def _fetch_track_metadata(sp, track_ids: List[str]) -> Dict[str, Dict]:
    meta: Dict[str, Dict] = {}
    if sp is None or not track_ids:
        return meta
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i : i + 50]
        resp = sp.tracks(batch)
        for t in resp.get("tracks", []):
            tid = t.get("id")
            album = t.get("album", {})
            images = album.get("images", [])
            image_url = images[0]["url"] if images else None
            meta[tid] = {
                "image_url": image_url,
                "preview_url": t.get("preview_url"),
                "explicit": bool(t.get("explicit")),
            }
    return meta


def _kv_get(conn: sqlite3.Connection, user_id: str, key: str) -> Optional[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS app_kv (user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, PRIMARY KEY(user_id, key));"
    )
    row = conn.execute(
        "SELECT value FROM app_kv WHERE user_id=? AND key=?;", (user_id, key)
    ).fetchone()
    return row[0] if row else None


def _kv_set(conn: sqlite3.Connection, user_id: str, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO app_kv(user_id, key, value) VALUES(?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value;",
        (user_id, key, value),
    )


st.set_page_config(page_title="Song of the Day", page_icon="🎵", layout="centered")

st.title("Song of the Day 🎵")

# Sidebar settings
with st.sidebar:
    st.markdown("**Settings**")
    use_data_client = st.toggle("Use new data client", value=False, help="Switch between legacy and new client")
    avoid_explicit = st.checkbox("Avoid explicit content", value=False)

# Connect block
if "connected" not in st.session_state:
    st.session_state.connected = False
    st.session_state.user_id = None
    st.session_state.client = None
    st.session_state.sp = None

if st.button("Connect to Spotify", type="primary") or st.session_state.connected:
    try:
        client, user_id, sp = _get_sp_client(use_data_client)
        st.session_state.client = client
        st.session_state.user_id = user_id
        st.session_state.sp = sp
        st.session_state.connected = True
        st.success(f"Connected as {user_id}")
    except Exception as e:
        st.error(f"Spotify authentication failed: {e}")

if not st.session_state.connected:
    st.info("Click 'Connect to Spotify' to authorize the app.")
    st.stop()

client = st.session_state.client
user_id = st.session_state.user_id
sp = st.session_state.sp

# Mood selection
lexicon = list(mood_to_feature_targets.__globals__["_MOOD_ANCHORS"].keys())  # type: ignore[attr-defined]

settings = load_settings()
conn = sqlite3.connect(settings.db_path)
with conn:
    last_mood = _kv_get(conn, user_id, "last_mood")

def _best_match(text: str, choices: List[str]) -> Optional[str]:
    t = text.strip().lower()
    if not t:
        return None
    if t in choices:
        return t
    # simple fuzzy: choose by max token overlap
    best = None
    best_score = -1
    for c in choices:
        score = len(set(t.split()) & set(c.split()))
        if score > best_score:
            best_score = score
            best = c
    return best

col1, col2 = st.columns([2, 3], gap="large")
with col1:
    mood_choice = st.selectbox("Mood preset (optional)", options=["(none)"] + lexicon, index=(lexicon.index(last_mood) + 1) if last_mood in lexicon else 0)
with col2:
    mood_free = st.text_input("Or type a mood", value=last_mood or "", placeholder="happy / calm / melancholic / hype …")

final_mood: Optional[str] = None
if mood_choice != "(none)":
    final_mood = mood_choice
elif mood_free:
    m = _best_match(mood_free, lexicon)
    final_mood = m

# Actions
if st.button("Recommend", type="primary"):
    with st.spinner("Generating candidates and computing scores…"):
        # Candidate pool
        cands = generate_candidate_pool(client, mood=final_mood)

        # Fetch recent and features for profile
        recent_df, recent_feats_df = _fetch_recent_and_features(client)

        # Add explicit flag via metadata
        meta = _fetch_track_metadata(sp, cands["track_id"].dropna().tolist()) if not cands.empty else {}
        if meta:
            cands["explicit"] = cands["track_id"].map(lambda tid: bool(meta.get(tid, {}).get("explicit")))

        # Rerank
        ranked = rerank_candidates(
            candidates_df=cands,
            recent_tracks_df=recent_df,
            recent_audio_features_df=recent_feats_df,
            mood=final_mood,
            exclude_explicit=avoid_explicit,
            top_k=max(20, 6),
        )

        # Persist last mood
        with conn:
            _kv_set(conn, user_id, "last_mood", final_mood or "")

        # Deterministic daily pick
        result = pick_song_of_day(user_id=user_id, ranked_results=ranked)

    # Render UI
    sotd = result.get("song_of_day")
    alternates = result.get("alternates", [])

    if not sotd:
        st.error("No recommendation produced.")
        st.stop()

    # Merge preview and cover
    all_ids = [sotd["id"]] + [a["id"] for a in alternates]
    meta_all = _fetch_track_metadata(sp, all_ids)

    def _art(url: Optional[str]) -> str:
        return url or ""

    # Big card
    st.subheader("Song of the Day")
    cover = meta_all.get(sotd["id"], {}).get("image_url")
    cols = st.columns([1, 2])
    with cols[0]:
        if cover:
            st.image(cover, width=200)
    with cols[1]:
        st.markdown(f"**{sotd['name']}** — {sotd['artist']}")
        if sotd.get("preview_url") or meta_all.get(sotd["id"], {}).get("preview_url"):
            st.audio(sotd.get("preview_url") or meta_all.get(sotd["id"], {}).get("preview_url"))
        # Rationale bullets
        rationale = sotd.get("rationale") or ""
        for bit in [b.strip() for b in rationale.split(",") if b.strip()]:
            st.write(f"- {bit}")

    # Show why toggle
    show_why = st.toggle("Show why (scores and features)")
    if show_why:
        # Locate detailed scores from ranked list
        detail_map = {r.get("track_id"): r for r in ranked}
        s_detail = detail_map.get(sotd["id"], {})
        st.caption(
            f"novelty={s_detail.get('novelty', ''):.2f} · fit={s_detail.get('fit_score', ''):.2f} · mood={s_detail.get('mood_score', ''):.2f}"
        )

    st.divider()
    st.subheader("Alternates")
    for alt in alternates:
        a_meta = meta_all.get(alt["id"], {})
        cols = st.columns([1, 3])
        with cols[0]:
            if a_meta.get("image_url"):
                st.image(a_meta["image_url"], width=100)
        with cols[1]:
            st.markdown(f"**{alt['name']}** — {alt['artist']}")
            if a_meta.get("preview_url"):
                st.audio(a_meta["preview_url"]) 
            if alt.get("rationale"):
                st.caption(alt["rationale"]) 

st.markdown("---")
st.markdown("To run: `poetry run streamlit run app.py`")

