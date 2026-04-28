## PROJECT NAME
## Tokslaw Vibe 

- Explicitly name your original project (from Modules 1-3) and provide a 2-3 sentence summary of its original goals and capabilities.

## Title and Summary
What your project does and why it matters.
This Music model suggests 10 songs from the dataset based on a user's preferred genre, valence, acoustiness and energy level.

---
## Core Features

### 1. Content-Based Recommendation Engine
- Scores songs using Euclidean distance on normalized audio features (energy, valence, acousticness)
- Genre-exact-match bonus ensures recommendations align with user taste
- Scales to 114k songs with real-time scoring

### 2. Context-Aware Adaptations
- **Activity Contexts**: studying (low energy, acoustic), workout (high energy, danceable), commute (balanced)
- **Time-of-Day Adjustments**: morning (uplifting), evening (calm, acoustic)
- **Explicit Content Filtering**: user-controlled tolerance levels

### 3. Recommendation Documentation & Validation
- **`explain_recommendation()`**: Detailed breakdown for each song (genre match, feature alignment, context bonuses)
- **`validate_recommendations()`**: Automated quality checks (genre match rate, explicit content compliance, feature distance)
- **`get_recommendation_documentation()`**: Full report with explanations, metrics, and validation results

**Key Metrics Validated**:
- Genre match rate (target: ≥30%)
- Explicit content rate respects user tolerance
- Average feature distance should be <0.8 (closer = better alignment)
- Overall recommendation score (0-1, higher is better)
### Architecture Overview: A short explanation of your system diagram.



```
User Profile (preferences + context)
		 ↓
Contextual Preference Adjuster (modifies targets based on activity/time)
		 ↓
Feature Distance Calculator (Euclidean distance on [energy, valence, acousticness])
		 ↓
Genre Matcher + Context Scorer (bonuses for genre match, instrumentalness, danceability)
		 ↓
Final Score (0-1) per song
		 ↓
Top-k Rankings + Explanations + Validation Report
```
---
## Setup Instructions: 
- Step-by-step directions to run your code.

1. Load songs: `songs = load_songs('data/dataset.csv')`
2. Create recommender: `rec = Recommender(songs)`
3. Define user profile: `user = UserProfile(favorite_genre='acoustic', target_energy=0.8, ...)`
4. Get recommendations: `recs = rec.recommend(user, k=10)`
5. Generate documentation: `print(rec.get_recommendation_documentation(user, k=10))`
6. Validate results: `validation = rec.validate_recommendations(user, recs)`
- Include at least 2-3 examples of inputs and the resulting AI outputs to demonstrate the system is functional.


**Example 1: Studying Session**
```python
user = UserProfile(
	favorite_genre='acoustic',
	target_energy=0.7, target_valence=0.6, preferred_acousticness=0.7,
	activity_context='studying', time_of_day='evening', explicit_tolerance=0.0
)
recommendations = rec.get_recommendation_documentation(user, k=5)
```
Output: 5 calm, acoustic, non-explicit songs matched to your study session.

**Example 2: Workout Playlist**
```python
user = UserProfile(
	favorite_genre='pop',
	target_energy=0.7, target_valence=0.6, preferred_acousticness=0.1,
	activity_context='workout', time_of_day='morning', explicit_tolerance=1.0
)
recommendations = rec.get_recommendation_documentation(user, k=10)
```
Output: 10 high-energy, danceable pop songs with validation confirming all match preferences.

**Example 3: Commute with Quality Assurance**
```python
user = UserProfile(favorite_genre='electronic', activity_context='commute')
recs = rec.recommend(user, k=5)
report = rec.get_recommendation_documentation(user, k=5)
validation = rec.validate_recommendations(user, recs)
print(f"Quality: {validation['valid']}, Genre match: {validation['genre_match_rate']:.0%}")
```
Output: Full report + validation metrics ensure recommendations are high-quality.
## Design Decisions: 
- Why you built it this way, and what trade-offs you made.
 

1. **Euclidean Distance Over ML Models**: Simple, interpretable, and fast for real-time recommendations. Trade-off: less nuanced than collaborative filtering.
2. **Context Adjustments as Preference Shifts**: Instead of separate ranking, we adjust user targets based on context. This keeps the score function consistent.
3. **Validation as Separate Layer**: Allows users to check quality without modifying the scoring logic. Metrics are conservative (stricter = safer).
4. **Feature Selection**: Used energy, valence, acousticness as core features—high variance, user-friendly, already normalized in dataset.
---
## Testing Summary: 
- What worked, what didn't, and what you learned.


**What Worked**:
- Explicit filtering perfectly blocks unwanted content (0% rate when tolerance=0.0)
- Context-aware scoring shows clear score deltas between study/workout/commute modes
- Validation catches low genre-match scenarios and alerts users
- Fast even on 114k songs (sub-second recommendation generation)

**Edge Cases Handled**:
- Empty recommendations: returns early with "No recommendations provided"
- Extreme feature mismatches: clamped to [0, 1] score range
- Users with very narrow preferences: validation warns if genre match <30%

**Lessons Learned**:
- Feature normalization is critical—mixing raw tempo (BPM) with 0-1 features broke distance calculations
- Context bonuses need caps (≤1.05) to avoid inflating scores above baseline quality
- Validation is user-friendly when it explains "why" not just "pass/fail"
**Retrieval-Augmented Generation (RAG)**
- retrieval of external api and automated validation of answers added

---
***What this project taught about AI and problem-solving***

This project shows the importance of **explainability and validation** in recommendation systems:
1. A black-box score is useless—users need to know *why* they got a recommendation.
2. Simple, geometric approaches (Euclidean distance) are often better than complex models if they're interpretable and controllable.
3. Context matters enormously—the same song is "good" or "bad" depending on *when* and *why* you're listening.
4. Validation metrics must match real constraints (explicit content, genre diversity) not just numerical accuracy.
***What this project taught you about AI and problem-solving.***

### Functionality
**Music Recommender System** — A content-based recommendation engine that suggests songs tailored to user preferences and listening context.

**Core features:**
- **Content similarity scoring:** Compares songs across numerical features (energy, valence, acousticness) and genre to find matches closest to the user's taste
- **Listening context adaptation:** Adjusts recommendations for situations like studying (lower energy, more acoustic), workouts (high energy, danceable), or commutes (balanced)
- **Explicit content filtering:** Allows users to control tolerance for explicit lyrics based on the listening context
- **Scalable scoring:** Loads 114k songs and ranks them in real time using Euclidean distance on normalized audio features

**How it works:**
1. User defines base preferences (favorite genre, target energy, valence, acousticness)
2. Optionally adds context (activity, time of day, explicit tolerance)
3. Scorer adjusts the target profile for that context, calculates distance to each song, and applies a bonus/penalty multiplier
4. Returns top-k songs sorted by final score (0–1)

The system prioritizes simplicity—no collaborative filtering or neural networks—just geometric distance in feature space, making it interpretable and fast.