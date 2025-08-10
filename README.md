## Song of the Day (Spotify)

A small recommender that selects a single "Song of the Day" plus five alternates using your Spotify data. It balances novelty (penalizes repeats), user fit, optional mood match (via valence–arousal mapping), and diversity.

### Stack
- Python 3.11, Poetry
- spotipy, pandas, numpy, scikit-learn
- streamlit for a quick UI
- SQLite for lightweight caching

### Features
- Pulls recent tracks and top artists/genres
- Computes novelty score using recent history and lifetime plays
- Optional mood input mapped to audio features (valence–arousal)
- Generates candidates and re-ranks for novelty, fit, mood, and diversity
- Outputs a single pick + 5 alternates with rationale

### Setup
1. Install Poetry:
   - macOS: `brew install poetry`
   - or: `curl -sSL https://install.python-poetry.org | python3 -`
2. Create your `.env`:
   - `cp example.env .env`
   - Fill in `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`, `SPOTIFY_USERNAME`.
3. Install dependencies:
   - `poetry install`
4. Activate the virtualenv (optional):
   - `poetry shell`

### CLI
Run the recommender from terminal:

```bash
poetry run song-of-the-day --mood "chill" --limit 20
```

### Streamlit App
Quick UI:

```bash
poetry run streamlit run app/streamlit_app.py
```

### Notes
- Ensure your Spotify app Redirect URI matches `.env` (`http://localhost:8080/callback` by default).
- The SQLite cache is stored at `./data/sotd_cache.db` (created on first run).
