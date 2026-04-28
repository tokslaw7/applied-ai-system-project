
from pathlib import Path

from .recommender import load_songs, recommend_songs


BASE_DIR = Path(__file__).resolve().parent.parent

def main() -> None:
    songs = load_songs(str(BASE_DIR / "data" / "dataset.csv"))

    # Starter example profile
    # user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    #user_prefs = {"genre": "acoustic", "energy": 0.35, "activity_context": "studying", "time_of_day": "evening"}

    user_prefs = {"genre": "pop", "energy": 0.7, "valence": 0.6, "activity_context": "workout", "time_of_day": "morning"}

    recommendations = recommend_songs(user_prefs, songs, k=10)
    #recommendations2 = recommend_songs(user_prefs2, songs, k=5) 

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()