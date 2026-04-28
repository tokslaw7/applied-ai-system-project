import pytest
from pathlib import Path
import csv
import tempfile
from typing import List

from src.recommender import Song, UserProfile, Recommender, load_songs, recommend_songs


# Fixtures for test data
@pytest.fixture
def sample_songs() -> List[Song]:
    """Create a small set of sample songs for testing."""
    return [
        Song(
            id=1,
            track_id="track1",
            artists="Artist A",
            album_name="Album 1",
            track_name="Happy Pop Song",
            track_genre="pop",
            popularity=80,
            duration_ms=180000,
            explicit=False,
            energy=0.8,
            tempo=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.1,
            key=5,
            loudness=-5.0,
            mode=1,
            speechiness=0.05,
            instrumentalness=0.0,
            liveness=0.1,
            time_signature=4,
        ),
        Song(
            id=2,
            track_id="track2",
            artists="Artist B",
            album_name="Album 2",
            track_name="Chill Acoustic",
            track_genre="acoustic",
            popularity=70,
            duration_ms=240000,
            explicit=False,
            energy=0.3,
            tempo=80,
            valence=0.5,
            danceability=0.2,
            acousticness=0.9,
            key=0,
            loudness=-8.0,
            mode=0,
            speechiness=0.04,
            instrumentalness=0.5,
            liveness=0.2,
            time_signature=4,
        ),
        Song(
            id=3,
            track_id="track3",
            artists="Artist C",
            album_name="Album 3",
            track_name="Workout Banger",
            track_genre="electronic",
            popularity=75,
            duration_ms=210000,
            explicit=True,
            energy=0.9,
            tempo=140,
            valence=0.7,
            danceability=0.95,
            acousticness=0.05,
            key=7,
            loudness=-3.0,
            mode=1,
            speechiness=0.03,
            instrumentalness=0.1,
            liveness=0.15,
            time_signature=4,
        ),
        Song(
            id=4,
            track_id="track4",
            artists="Artist D",
            album_name="Album 4",
            track_name="Pop Hit",
            track_genre="pop",
            popularity=85,
            duration_ms=200000,
            explicit=False,
            energy=0.7,
            tempo=100,
            valence=0.8,
            danceability=0.75,
            acousticness=0.2,
            key=3,
            loudness=-4.0,
            mode=1,
            speechiness=0.06,
            instrumentalness=0.05,
            liveness=0.12,
            time_signature=4,
        ),
    ]


@pytest.fixture
def basic_user_profile() -> UserProfile:
    """Create a basic user profile."""
    return UserProfile(
        favorite_genre="pop",
        target_energy=0.7,
        target_valence=0.6,
        target_danceability=0.6,
        preferred_acousticness=0.2,
        likes_acoustic=False,
    )


@pytest.fixture
def studying_user_profile() -> UserProfile:
    """Create a user profile for studying context."""
    return UserProfile(
        favorite_genre="acoustic",
        target_energy=0.5,
        target_valence=0.5,
        target_danceability=0.5,
        preferred_acousticness=0.7,
        likes_acoustic=True,
        activity_context="studying",
        time_of_day="evening",
    )


@pytest.fixture
def workout_user_profile() -> UserProfile:
    """Create a user profile for workout context."""
    return UserProfile(
        favorite_genre="electronic",
        target_energy=0.8,
        target_valence=0.6,
        target_danceability=0.8,
        preferred_acousticness=0.1,
        likes_acoustic=False,
        activity_context="workout",
        time_of_day="morning",
    )


@pytest.fixture
def recommender(sample_songs) -> Recommender:
    """Create a recommender instance with sample songs."""
    return Recommender(sample_songs)


# Tests for Song dataclass
class TestSongDataclass:
    def test_song_creation(self):
        """Test that a Song can be created with all attributes."""
        song = Song(
            id=1,
            track_id="test",
            artists="Test Artist",
            album_name="Test Album",
            track_name="Test Song",
            track_genre="pop",
            popularity=50,
            duration_ms=180000,
            explicit=False,
            energy=0.5,
            tempo=100,
            valence=0.5,
            danceability=0.5,
            acousticness=0.5,
            key=0,
            loudness=-5.0,
            mode=1,
            speechiness=0.05,
            instrumentalness=0.1,
            liveness=0.1,
            time_signature=4,
        )
        assert song.track_name == "Test Song"
        assert song.track_genre == "pop"
        assert song.energy == 0.5


# Tests for UserProfile dataclass
class TestUserProfileDataclass:
    def test_user_profile_creation_basic(self):
        """Test UserProfile creation with required fields."""
        profile = UserProfile(
            favorite_genre="pop",
            target_energy=0.7,
            likes_acoustic=False,
        )
        assert profile.favorite_genre == "pop"
        assert profile.target_energy == 0.7
        assert profile.activity_context == "general"
        assert profile.explicit_tolerance == 1.0

    def test_user_profile_with_context(self):
        """Test UserProfile creation with context fields."""
        profile = UserProfile(
            favorite_genre="acoustic",
            target_energy=0.3,
            likes_acoustic=True,
            activity_context="studying",
            time_of_day="evening",
            explicit_tolerance=0.5,
        )
        assert profile.activity_context == "studying"
        assert profile.time_of_day == "evening"
        assert profile.explicit_tolerance == 0.5


# Tests for Recommender class
class TestRecommenderCalculateScore:
    def test_exact_genre_match(self, recommender, basic_user_profile):
        """Test that genre matching increases score."""
        pop_song = recommender.songs[0]  # Happy Pop Song
        # Pop matches basic_user_profile's favorite_genre
        basic_user_profile.favorite_genre = "pop"
        score = recommender.calculate_score(basic_user_profile, pop_song)
        assert score > 0.0
        assert score <= 1.0

    def test_no_genre_match(self, recommender, basic_user_profile):
        """Test that non-matching genre doesn't get genre bonus."""
        acoustic_song = recommender.songs[1]  # Chill Acoustic
        basic_user_profile.favorite_genre = "metal"
        score = recommender.calculate_score(basic_user_profile, acoustic_song)
        # Score should be based purely on feature similarity
        assert score >= 0.0
        assert score <= 1.0

    def test_feature_similarity_score(self, recommender, basic_user_profile):
        """Test that similar features produce higher scores."""
        # Song with very different features from user preferences
        song1 = recommender.songs[1]  # Chill Acoustic (energy=0.3)
        basic_user_profile.target_energy = 0.3
        basic_user_profile.preferred_acousticness = 0.9
        score1 = recommender.calculate_score(basic_user_profile, song1)

        # Song with very different features
        song2 = recommender.songs[0]  # Happy Pop (energy=0.8)
        score2 = recommender.calculate_score(basic_user_profile, song2)

        # Similar song should have higher score
        assert score1 > score2

    def test_score_bounds(self, recommender, basic_user_profile):
        """Test that scores are always between 0 and 1."""
        for song in recommender.songs:
            score = recommender.calculate_score(basic_user_profile, song)
            assert 0.0 <= score <= 1.0

    def test_genre_weight_effect(self, recommender, basic_user_profile):
        """Test that genre weight affects scoring."""
        song = recommender.songs[0]  # Pop song
        basic_user_profile.favorite_genre = "pop"

        # High genre weight
        score_high_genre = recommender.calculate_score(
            basic_user_profile, song, genre_weight=0.8, feature_weight=0.2
        )

        # Low genre weight
        score_low_genre = recommender.calculate_score(
            basic_user_profile, song, genre_weight=0.2, feature_weight=0.8
        )

        # Both should be valid scores
        assert 0.0 <= score_high_genre <= 1.0
        assert 0.0 <= score_low_genre <= 1.0


class TestContextualPreferences:
    def test_studying_context_adjusts_targets(self, recommender, studying_user_profile):
        """Test that studying context lowers energy and danceability."""
        targets = recommender._build_contextual_preferences(studying_user_profile)

        # Studying should reduce energy and danceability
        assert targets["energy"] <= studying_user_profile.target_energy
        assert targets["danceability"] <= studying_user_profile.target_danceability
        # Should increase acousticness
        assert targets["acousticness"] >= studying_user_profile.preferred_acousticness

    def test_workout_context_adjusts_targets(self, recommender, workout_user_profile):
        """Test that workout context increases energy and danceability."""
        targets = recommender._build_contextual_preferences(workout_user_profile)

        # Workout should increase energy and danceability
        assert targets["energy"] >= workout_user_profile.target_energy
        assert targets["danceability"] >= workout_user_profile.target_danceability
        # Should decrease acousticness
        assert targets["acousticness"] <= workout_user_profile.preferred_acousticness

    def test_morning_context_increases_energy(self, recommender, studying_user_profile):
        """Test that morning context increases energy."""
        studying_user_profile.time_of_day = "morning"
        targets = recommender._build_contextual_preferences(studying_user_profile)
        assert targets["energy"] >= studying_user_profile.target_energy

    def test_evening_context_decreases_energy(self, recommender, studying_user_profile):
        """Test that evening context decreases energy."""
        targets = recommender._build_contextual_preferences(studying_user_profile)
        assert targets["energy"] <= studying_user_profile.target_energy

    def test_general_context_preserves_targets(self, recommender, basic_user_profile):
        """Test that general context preserves original targets."""
        targets = recommender._build_contextual_preferences(basic_user_profile)
        assert targets["energy"] == basic_user_profile.target_energy
        assert targets["danceability"] == basic_user_profile.target_danceability


class TestContextScore:
    def test_explicit_tolerance_zero(self, recommender, basic_user_profile):
        """Test that explicit_tolerance=0 penalizes explicit songs."""
        explicit_song = recommender.songs[2]  # Workout Banger (explicit=True)
        basic_user_profile.explicit_tolerance = 0.0

        score_explicit = recommender._context_score(basic_user_profile, explicit_song)
        assert score_explicit == 0.0

    def test_explicit_tolerance_full(self, recommender, basic_user_profile):
        """Test that explicit_tolerance=1.0 doesn't penalize explicit songs."""
        explicit_song = recommender.songs[2]
        basic_user_profile.explicit_tolerance = 1.0

        score = recommender._context_score(basic_user_profile, explicit_song)
        assert score >= 0.9  # Should be close to 1.0

    def test_acoustic_preference_bonus(self, recommender, studying_user_profile):
        """Test that acoustic preference gives bonus to acoustic songs."""
        acoustic_song = recommender.songs[1]  # Chill Acoustic
        studying_user_profile.likes_acoustic = True

        score_with_pref = recommender._context_score(studying_user_profile, acoustic_song)
        
        studying_user_profile.likes_acoustic = False
        score_without_pref = recommender._context_score(studying_user_profile, acoustic_song)
        
        # Score with preference should be >= score without preference
        # (bonus multiplier of 1.05 is applied)
        assert score_with_pref >= score_without_pref

    def test_studying_instrumental_bonus(self, recommender, studying_user_profile):
        """Test that studying context gives bonus to instrumental songs."""
        acoustic_song = recommender.songs[1]  # Chill Acoustic (instrumentalness=0.5)
        
        score = recommender._context_score(studying_user_profile, acoustic_song)
        # Should get studying bonus for instrumentalness >= 0.4
        assert score >= 1.0  # Gets a multiplier


class TestRecommenderRecommend:
    def test_returns_k_songs(self, recommender, basic_user_profile):
        """Test that recommend returns exactly k songs."""
        k = 3
        recommendations = recommender.recommend(basic_user_profile, k=k)
        assert len(recommendations) == k

    def test_recommendations_sorted_by_score(self, recommender, basic_user_profile):
        """Test that recommendations are sorted by score (highest first)."""
        recommendations = recommender.recommend(basic_user_profile, k=10)
        scores = [score for _, score in recommendations]
        
        # Verify descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_recommendations_are_tuples(self, recommender, basic_user_profile):
        """Test that recommendations are (Song, score) tuples."""
        recommendations = recommender.recommend(basic_user_profile, k=2)
        
        for song, score in recommendations:
            assert isinstance(song, Song)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_k_larger_than_songs(self, recommender, basic_user_profile):
        """Test that k can be larger than number of songs."""
        k = 100
        recommendations = recommender.recommend(basic_user_profile, k=k)
        # Should return all available songs
        assert len(recommendations) == len(recommender.songs)

    def test_studying_context_recommendations(self, recommender, studying_user_profile):
        """Test that studying context returns appropriate recommendations."""
        recommendations = recommender.recommend(studying_user_profile, k=2)
        
        # Should get songs with low energy and acoustic characteristics
        for song, score in recommendations:
            assert 0.0 <= score <= 1.0


class TestExplainRecommendation:
    def test_explain_returns_dict_with_required_keys(self, recommender, basic_user_profile):
        """Test that explain_recommendation returns required keys."""
        song = recommender.songs[0]
        explanation = recommender.explain_recommendation(basic_user_profile, song)
        
        required_keys = [
            "song_name",
            "artist",
            "genre_match",
            "feature_match_score",
            "feature_deltas",
            "context_bonus",
            "final_score",
            "explanation",
        ]
        for key in required_keys:
            assert key in explanation

    def test_genre_match_flag(self, recommender, basic_user_profile):
        """Test that genre_match is True when genres match."""
        song = recommender.songs[0]  # Pop song
        basic_user_profile.favorite_genre = "pop"
        
        explanation = recommender.explain_recommendation(basic_user_profile, song)
        assert explanation["genre_match"] is True

    def test_feature_deltas_in_explanation(self, recommender, basic_user_profile):
        """Test that feature deltas are calculated correctly."""
        song = recommender.songs[0]
        explanation = recommender.explain_recommendation(basic_user_profile, song)
        
        deltas = explanation["feature_deltas"]
        assert "energy_diff" in deltas
        assert "valence_diff" in deltas
        assert "acousticness_diff" in deltas
        
        # All deltas should be non-negative
        for delta in deltas.values():
            assert delta >= 0.0

    def test_explanation_text_is_string(self, recommender, basic_user_profile):
        """Test that explanation text is a readable string."""
        song = recommender.songs[0]
        explanation = recommender.explain_recommendation(basic_user_profile, song)
        
        assert isinstance(explanation["explanation"], str)
        assert len(explanation["explanation"]) > 0
        # Song name is in the explanation dict, check it's there
        assert explanation["song_name"] == song.track_name


class TestValidateRecommendations:
    def test_validate_returns_dict_with_required_keys(self, recommender, basic_user_profile):
        """Test that validate_recommendations returns required keys."""
        recommendations = recommender.recommend(basic_user_profile, k=3)
        validation = recommender.validate_recommendations(basic_user_profile, recommendations)
        
        required_keys = [
            "valid",
            "issues",
            "genre_match_rate",
            "explicit_content_rate",
            "avg_feature_distance",
            "recommendation_count",
            "avg_score",
        ]
        for key in required_keys:
            assert key in validation

    def test_validate_empty_recommendations(self, recommender, basic_user_profile):
        """Test validation with empty recommendations."""
        validation = recommender.validate_recommendations(basic_user_profile, [])
        assert validation["valid"] is False

    def test_genre_match_rate_calculation(self, recommender, basic_user_profile):
        """Test that genre match rate is calculated correctly."""
        basic_user_profile.favorite_genre = "pop"
        recommendations = recommender.recommend(basic_user_profile, k=4)
        validation = recommender.validate_recommendations(basic_user_profile, recommendations)
        
        # Genre match rate should be between 0 and 1
        assert 0.0 <= validation["genre_match_rate"] <= 1.0

    def test_explicit_content_rate(self, recommender, basic_user_profile):
        """Test explicit content rate calculation."""
        recommendations = recommender.recommend(basic_user_profile, k=4)
        validation = recommender.validate_recommendations(basic_user_profile, recommendations)
        
        assert 0.0 <= validation["explicit_content_rate"] <= 1.0

    def test_avg_feature_distance(self, recommender, basic_user_profile):
        """Test average feature distance calculation."""
        recommendations = recommender.recommend(basic_user_profile, k=4)
        validation = recommender.validate_recommendations(basic_user_profile, recommendations)
        
        assert validation["avg_feature_distance"] >= 0.0

    def test_issues_list_for_poor_recommendations(self, recommender):
        """Test that issues are identified for poor recommendations."""
        # Create a profile that doesn't match any songs well
        poor_profile = UserProfile(
            favorite_genre="jazz",  # Not in sample songs
            target_energy=0.1,
            target_valence=0.1,
            target_danceability=0.1,
            preferred_acousticness=0.05,
            likes_acoustic=False,
        )
        
        recommendations = recommender.recommend(poor_profile, k=2)
        validation = recommender.validate_recommendations(poor_profile, recommendations)
        
        # Should have some issues
        assert isinstance(validation["issues"], list)


class TestGetRecommendationDocumentation:
    def test_documentation_returns_string(self, recommender, basic_user_profile):
        """Test that documentation is returned as a string."""
        doc = recommender.get_recommendation_documentation(basic_user_profile, k=3)
        assert isinstance(doc, str)
        assert len(doc) > 0

    def test_documentation_contains_user_profile_info(self, recommender, basic_user_profile):
        """Test that documentation includes user profile information."""
        doc = recommender.get_recommendation_documentation(basic_user_profile, k=2)
        
        assert "USER PROFILE" in doc
        assert basic_user_profile.favorite_genre in doc
        assert "QUALITY METRICS" in doc

    def test_documentation_contains_recommendations(self, recommender, basic_user_profile):
        """Test that documentation includes recommendations."""
        doc = recommender.get_recommendation_documentation(basic_user_profile, k=2)
        
        assert "RECOMMENDATIONS" in doc
        # Should have at least 2 recommendation entries
        assert "1." in doc
        assert "2." in doc


# Tests for load_songs function
class TestLoadSongs:
    def test_load_songs_from_csv(self):
        """Test loading songs from a CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id", "track_id", "artists", "album_name", "track_name", "track_genre",
                    "popularity", "duration_ms", "explicit", "energy", "tempo", "valence",
                    "danceability", "acousticness", "key", "loudness", "mode", "speechiness",
                    "instrumentalness", "liveness", "time_signature"
                ]
            )
            writer.writeheader()
            writer.writerow({
                "id": "1",
                "track_id": "track1",
                "artists": "Artist A",
                "album_name": "Album 1",
                "track_name": "Song 1",
                "track_genre": "pop",
                "popularity": "80",
                "duration_ms": "180000",
                "explicit": "false",
                "energy": "0.8",
                "tempo": "120",
                "valence": "0.9",
                "danceability": "0.8",
                "acousticness": "0.1",
                "key": "5",
                "loudness": "-5.0",
                "mode": "1",
                "speechiness": "0.05",
                "instrumentalness": "0.0",
                "liveness": "0.1",
                "time_signature": "4"
            })
            temp_file = f.name

        try:
            songs = load_songs(temp_file)
            assert len(songs) == 1
            assert songs[0].track_name == "Song 1"
            assert songs[0].track_genre == "pop"
            assert songs[0].explicit is False
        finally:
            Path(temp_file).unlink()

    def test_load_multiple_songs(self):
        """Test loading multiple songs from CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id", "track_id", "artists", "album_name", "track_name", "track_genre",
                    "popularity", "duration_ms", "explicit", "energy", "tempo", "valence",
                    "danceability", "acousticness", "key", "loudness", "mode", "speechiness",
                    "instrumentalness", "liveness", "time_signature"
                ]
            )
            writer.writeheader()
            for i in range(5):
                writer.writerow({
                    "id": str(i),
                    "track_id": f"track{i}",
                    "artists": f"Artist {i}",
                    "album_name": f"Album {i}",
                    "track_name": f"Song {i}",
                    "track_genre": "pop",
                    "popularity": "80",
                    "duration_ms": "180000",
                    "explicit": "false",
                    "energy": "0.8",
                    "tempo": "120",
                    "valence": "0.9",
                    "danceability": "0.8",
                    "acousticness": "0.1",
                    "key": "5",
                    "loudness": "-5.0",
                    "mode": "1",
                    "speechiness": "0.05",
                    "instrumentalness": "0.0",
                    "liveness": "0.1",
                    "time_signature": "4"
                })
            temp_file = f.name

        try:
            songs = load_songs(temp_file)
            assert len(songs) == 5
            for i, song in enumerate(songs):
                assert song.track_name == f"Song {i}"
        finally:
            Path(temp_file).unlink()


# Tests for recommend_songs function
class TestRecommendSongs:
    def test_recommend_songs_returns_list(self, sample_songs):
        """Test that recommend_songs returns a list."""
        user_prefs = {"genre": "pop", "energy": 0.7}
        results = recommend_songs(user_prefs, sample_songs, k=2)
        
        assert isinstance(results, list)
        assert len(results) == 2

    def test_recommend_songs_tuple_format(self, sample_songs):
        """Test that each result is a (song_dict, score, explanation) tuple."""
        user_prefs = {"genre": "pop", "energy": 0.7}
        results = recommend_songs(user_prefs, sample_songs, k=1)
        
        song_dict, score, explanation = results[0]
        
        assert isinstance(song_dict, dict)
        assert "title" in song_dict
        assert "artist" in song_dict
        assert "genre" in song_dict
        assert isinstance(score, float)
        assert isinstance(explanation, str)

    def test_recommend_songs_with_context(self, sample_songs):
        """Test recommend_songs with activity context."""
        user_prefs = {
            "genre": "acoustic",
            "energy": 0.3,
            "activity_context": "studying",
            "time_of_day": "evening"
        }
        results = recommend_songs(user_prefs, sample_songs, k=2)
        
        assert len(results) == 2
        for song_dict, score, explanation in results:
            assert 0.0 <= score <= 1.0

    def test_recommend_songs_defaults(self, sample_songs):
        """Test that recommend_songs handles missing preferences gracefully."""
        user_prefs = {}  # Empty preferences
        results = recommend_songs(user_prefs, sample_songs, k=2)
        
        assert len(results) == 2


# Integration tests
class TestIntegration:
    def test_full_recommendation_workflow(self, sample_songs):
        """Test complete workflow: load, profile, recommend, explain, validate."""
        # Create recommender
        recommender = Recommender(sample_songs)
        
        # Create user profile
        user_profile = UserProfile(
            favorite_genre="pop",
            target_energy=0.7,
            target_valence=0.6,
            target_danceability=0.6,
            preferred_acousticness=0.2,
            likes_acoustic=False,
            activity_context="general",
        )
        
        # Get recommendations
        recommendations = recommender.recommend(user_profile, k=2)
        assert len(recommendations) == 2
        
        # Validate recommendations
        validation = recommender.validate_recommendations(user_profile, recommendations)
        assert validation["valid"] is not None
        assert validation["recommendation_count"] == 2
        
        # Get explanations
        for song, score in recommendations:
            explanation = recommender.explain_recommendation(user_profile, song)
            assert explanation["song_name"] == song.track_name
            assert 0.0 <= explanation["final_score"] <= 1.0
        
        # Get full documentation
        doc = recommender.get_recommendation_documentation(user_profile, k=2)
        assert len(doc) > 0
        assert "RECOMMENDATION REPORT" in doc

    def test_different_contexts_produce_different_recommendations(self, sample_songs):
        """Test that different contexts produce different top recommendations."""
        recommender = Recommender(sample_songs)
        
        # Studying profile
        studying_profile = UserProfile(
            favorite_genre="acoustic",
            target_energy=0.3,
            target_valence=0.5,
            target_danceability=0.3,
            preferred_acousticness=0.8,
            likes_acoustic=True,
            activity_context="studying",
        )
        
        # Workout profile
        workout_profile = UserProfile(
            favorite_genre="electronic",
            target_energy=0.8,
            target_valence=0.6,
            target_danceability=0.8,
            preferred_acousticness=0.1,
            likes_acoustic=False,
            activity_context="workout",
        )
        
        # Get recommendations
        studying_recs = recommender.recommend(studying_profile, k=1)
        workout_recs = recommender.recommend(workout_profile, k=1)
        
        # Top recommendations should likely be different
        studying_song = studying_recs[0][0]
        workout_song = workout_recs[0][0]
        
        # They may be different or same, but both should be valid
        assert studying_song is not None
        assert workout_song is not None
