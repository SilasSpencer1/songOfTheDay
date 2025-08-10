from __future__ import annotations

import streamlit as st

from song_of_the_day.pipeline import generate_recommendations

st.set_page_config(page_title="Song of the Day", page_icon="🎵", layout="centered")

st.title("Song of the Day 🎵")

mood = st.text_input("Mood (optional):", placeholder="happy | sad | chill | focus | party | angry")
limit = st.slider("Candidate pool size", min_value=10, max_value=100, value=20, step=5)

if st.button("Generate"):
    with st.spinner("Contacting Spotify and ranking candidates..."):
        result = generate_recommendations(mood_text=mood or None, rec_limit=limit)

    sotd = result.get("song_of_the_day")
    alternates = result.get("alternates", [])

    if not sotd:
        st.error(result.get("reason", "No recommendation produced."))
    else:
        st.subheader("Song of the Day")
        st.write(f"{sotd['track_name']} — {sotd['artist_names']}")
        st.caption(sotd["rationale"]) 

        st.subheader("Alternates")
        for a in alternates:
            st.write(f"{a['track_name']} — {a['artist_names']}")
            st.caption(a["rationale"]) 
