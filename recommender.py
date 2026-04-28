from typing import List, Dict, Tuple
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    track_id: str
    artists: str
    album_name: str
    track_name: str
    track_genre: str
    popularity: int
    duration_ms: int
    explicit: bool
    energy: float
    tempo: float
    valence: float
    danceability: float
    acousticness: float
    key: int    
    loudness: float
    mode: int
    speechiness: float
    instrumentalness: float
    liveness: float
    time_signature: int
  

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    target_energy: float
    likes_acoustic: bool
    preferred_tempo: float = 100.0
    target_valence: float = 0.5
    target_danceability: float = 0.5
    preferred_acousticness: float = 0.5
    activity_context: str = "general"
    time_of_day: str = "any"
    explicit_tolerance: float = 1.0

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _build_contextual_preferences(self, user_profile: UserProfile) -> Dict[str, float]:
        """
        Adjust the user's base preferences for listening context.

        Returns a normalized target profile that is used for scoring.
        """
        targets = {
            "energy": user_profile.target_energy,
            "valence": user_profile.target_valence,
            "danceability": user_profile.target_danceability,
            "acousticness": user_profile.preferred_acousticness,
            "tempo": user_profile.preferred_tempo,
        }

        activity_context = user_profile.activity_context.lower()
        if activity_context == "studying":
            targets["energy"] = min(targets["energy"], 0.35)
            targets["valence"] = min(targets["valence"], 0.55)
            targets["danceability"] = min(targets["danceability"], 0.45)
            targets["acousticness"] = max(targets["acousticness"], 0.55)
            targets["tempo"] = min(targets["tempo"], 110.0)
        elif activity_context == "workout":
            targets["energy"] = max(targets["energy"], 0.75)
            targets["valence"] = max(targets["valence"], 0.55)
            targets["danceability"] = max(targets["danceability"], 0.7)
            targets["acousticness"] = min(targets["acousticness"], 0.35)
            targets["tempo"] = max(targets["tempo"], 130.0)
        elif activity_context == "commute":
            targets["energy"] = min(max(targets["energy"], 0.45), 0.75)
            targets["valence"] = min(max(targets["valence"], 0.45), 0.75)
            targets["danceability"] = min(max(targets["danceability"], 0.5), 0.8)
            targets["tempo"] = min(max(targets["tempo"], 90.0), 140.0)

        time_of_day = user_profile.time_of_day.lower()
        if time_of_day == "morning":
            targets["energy"] = max(targets["energy"], 0.55)
            targets["valence"] = max(targets["valence"], 0.6)
            targets["tempo"] = max(targets["tempo"], 100.0)
        elif time_of_day == "afternoon":
            targets["energy"] = min(max(targets["energy"], 0.5), 0.8)
            targets["valence"] = min(max(targets["valence"], 0.45), 0.8)
        elif time_of_day == "evening":
            targets["energy"] = min(targets["energy"], 0.6)
            targets["valence"] = min(targets["valence"], 0.65)
            targets["acousticness"] = max(targets["acousticness"], 0.45)
            targets["tempo"] = min(targets["tempo"], 120.0)

        return targets

    def _context_score(self, user_profile: UserProfile, song: Song) -> float:
        """
        Score explicit-lyrics tolerance and short contextual fit signals.
        """
        score = 1.0

        if song.explicit:
            explicit_tolerance = max(0.0, min(1.0, user_profile.explicit_tolerance))
            score *= explicit_tolerance

        if user_profile.likes_acoustic and song.acousticness >= 0.5:
            score *= 1.05
        elif user_profile.likes_acoustic and song.acousticness < 0.2:
            score *= 0.9

        activity_context = user_profile.activity_context.lower()
        if activity_context == "studying" and song.instrumentalness >= 0.4:
            score *= 1.05
        elif activity_context == "workout" and song.danceability >= 0.7:
            score *= 1.05
        elif activity_context == "commute" and 0.35 <= song.valence <= 0.75:
            score *= 1.03

        return max(0.0, min(1.0, score))

    def calculate_score(self, user_profile: UserProfile, song: Song, 
                       genre_weight: float = 0.3, 
                       feature_weight: float = 0.7) -> float:
        """
        Score combines:
        - Genre matching (categorical): +1 if genres match, 0 otherwise
        - Numerical features (energy, valence, acousticness): based on Euclidean distance
        
        Args:
            user_profile: UserProfile with target preferences
            song: Song to score
            genre_weight: Weight for genre matching (0-1)
            feature_weight: Weight for numerical features (0-1)
        
        Returns:
            float: Similarity score between 0 and 1, where 1 is perfect match
        """
        # Genre score: exact match = 1.0, otherwise = 0.0
        genre_score = 1.0 if song.track_genre.lower() == user_profile.favorite_genre.lower() else 0.0
        
        # Numerical features score: based on Euclidean distance
        # Features to compare: energy, valence, acousticness
        contextual_targets = self._build_contextual_preferences(user_profile)
        user_features = [
            contextual_targets["energy"],
            contextual_targets["valence"],
            contextual_targets["acousticness"]
        ]
        song_features = [
            song.energy,
            song.valence,
            song.acousticness
        ]
        
        # Calculate Euclidean distance (all features are already normalized 0-1)
        euclidean_distance = sum((u - s) ** 2 for u, s in zip(user_features, song_features)) ** 0.5
        
        # Convert distance to similarity score (max distance in 3D normalized space is ~1.73)
        # Use inverse distance: closer songs get higher scores
        max_distance = (3 ** 0.5)  # Maximum possible distance in normalized 3D space
        feature_score = 1.0 - (euclidean_distance / max_distance)
        
        # Ensure score is between 0 and 1
        feature_score = max(0.0, min(1.0, feature_score))
        
        # Combine scores with weights
        # Note: weights should sum to 1.0 for normalized output
        total_weight = genre_weight + feature_weight
        combined_score = (genre_weight * genre_score + feature_weight * feature_score) / total_weight

        contextual_bonus = self._context_score(user_profile, song)
        combined_score = combined_score * contextual_bonus
        
        return combined_score

    def recommend(self, user_profile: UserProfile, k: int = 5,
                  genre_weight: float = 0.3, feature_weight: float = 0.7) -> List[Tuple[Song, float]]:
        """
        Get top k song recommendations for a user based on content similarity.
        
        Args:
            user_profile: UserProfile with target preferences
            k: Number of recommendations to return
            genre_weight: Weight for genre matching
            feature_weight: Weight for numerical features
        
        Returns:
            List of tuples: [(Song, score), ...] sorted by score descending
        """
        # Score all songs
        scored_songs = [
            (song, self.calculate_score(user_profile, song, genre_weight, feature_weight))
            for song in self.songs
        ]
        
        # Sort by score descending and return top k
        scored_songs.sort(key=lambda x: x[1], reverse=True)
        return scored_songs[:k]


def load_songs(file_path: str) -> List[Song]:
    """
    Loads songs from a CSV file and returns a list of Song objects.
    Required by tests/test_recommender.py
    """
    songs = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            song = Song(
                id=int(row['id']),
                track_id=row['track_id'],
                artists=row['artists'],
                album_name=row['album_name'],
                track_name=row['track_name'],
                track_genre=row['track_genre'],
                popularity=int(row['popularity']),
                duration_ms=int(row['duration_ms']),
                explicit=row['explicit'].lower() == 'true',
                energy=float(row['energy']),
                tempo=float(row['tempo']),
                valence=float(row['valence']),
                danceability=float(row['danceability']),
                acousticness=float(row['acousticness']),
                key=int(row['key']),
                loudness=float(row['loudness']),
                mode=int(row['mode']),
                speechiness=float(row['speechiness']),
                instrumentalness=float(row['instrumentalness']),
                liveness=float(row['liveness']),
                time_signature=int(row['time_signature'])
            )
            songs.append(song)
    return songs
