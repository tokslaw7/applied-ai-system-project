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
