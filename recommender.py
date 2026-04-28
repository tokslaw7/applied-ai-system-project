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

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

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
        user_features = [
            user_profile.target_energy,
            user_profile.target_valence,
            user_profile.preferred_acousticness
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
