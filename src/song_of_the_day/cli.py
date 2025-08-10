from __future__ import annotations

import argparse
import sys

from .pipeline import generate_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Song of the Day recommender (Spotify)")
    parser.add_argument("--mood", type=str, default=None, help="Optional mood: happy, sad, chill, focus, party, angry")
    parser.add_argument("--limit", type=int, default=20, help="Number of candidates to consider (re-ranking selects top 6)")
    args = parser.parse_args()

    result = generate_recommendations(mood_text=args.mood, rec_limit=args.limit)

    if not result.get("song_of_the_day"):
        print("No recommendation produced:", result.get("reason", "unknown"))
        sys.exit(1)

    print("Song of the Day:")
    s = result["song_of_the_day"]
    print(f"- {s['track_name']} — {s['artist_names']} ({s['rationale']})")
    print("\nAlternates:")
    for a in result["alternates"]:
        print(f"- {a['track_name']} — {a['artist_names']} ({a['rationale']})")


if __name__ == "__main__":
    main()
