# Test Suite Summary - Music Recommendation System

## Overview
Comprehensive test suite for the music recommendation system with **43 passing tests** covering all major components and edge cases.

## Test Coverage

### 1. **Dataclass Tests** (3 tests)
   - **Song dataclass creation**: Verify Song objects are created with all required attributes
   - **UserProfile basic creation**: Test UserProfile with only required fields
   - **UserProfile with context**: Test UserProfile with activity and time-of-day contexts

### 2. **Scoring Logic Tests** (8 tests)
   - **Exact genre matching**: Verify genre matches increase scores
   - **No genre match**: Test feature-based scoring without genre bonus
   - **Feature similarity**: Verify songs with similar features score higher
   - **Score bounds**: Ensure all scores are between 0 and 1
   - **Genre weight effects**: Test impact of genre_weight parameter
   - **Multiple context scenarios**: Test scoring across different user profiles

### 3. **Contextual Preferences Tests** (5 tests)
   - **Studying context**: Reduces energy, increases acousticness
   - **Workout context**: Increases energy and danceability
   - **Morning context**: Increases energy and valence
   - **Evening context**: Decreases energy and valence
   - **General context**: Preserves original target preferences

### 4. **Context Scoring Tests** (4 tests)
   - **Explicit tolerance=0**: Filters out explicit songs
   - **Explicit tolerance=1.0**: Doesn't penalize explicit songs
   - **Acoustic preference bonus**: Boosts acoustic songs when preferred
   - **Instrumental bonus for studying**: Rewards instrumental tracks for studying context

### 5. **Recommendation Engine Tests** (5 tests)
   - **Returns k songs**: Verify correct number of recommendations
   - **Sorted by score**: Ensure recommendations are ranked highest first
   - **Tuple format**: Validate (Song, score) structure
   - **K larger than dataset**: Handle requests for more songs than available
   - **Context-aware recommendations**: Test with different user contexts

### 6. **Recommendation Explanation Tests** (4 tests)
   - **Required keys**: Verify all necessary fields are in explanation dict
   - **Genre match flag**: Correctly identifies genre matches
   - **Feature deltas**: Calculates differences accurately
   - **Explanation text quality**: Generates readable explanations

### 7. **Validation Tests** (6 tests)
   - **Required validation keys**: Verify validation results contain all metrics
   - **Empty recommendations**: Handle edge case gracefully
   - **Genre match rate**: Correctly calculates percentage
   - **Explicit content rate**: Tracks explicit song percentage
   - **Average feature distance**: Measures recommendation quality
   - **Issues identification**: Flags poor recommendations

### 8. **Documentation Tests** (3 tests)
   - **Documentation format**: Returns properly formatted string report
   - **User profile info**: Includes user preferences in report
   - **Recommendation details**: Lists all recommendations with rankings

### 9. **CSV Loading Tests** (2 tests)
   - **Single song loading**: Read one song from CSV
   - **Multiple songs**: Load entire dataset from CSV

### 10. **Integration Tests** (5 tests)
   - **Full workflow**: Complete recommendation pipeline
   - **Top-level function**: Test recommend_songs() wrapper
   - **Context handling**: Different contexts produce different recommendations
   - **Error handling**: Graceful handling of missing preferences
   - **Format validation**: Output format consistency

## Key Test Fixtures

| Fixture | Purpose |
|---------|---------|
| `sample_songs` | 4 diverse test songs (pop, acoustic, electronic) |
| `basic_user_profile` | Standard pop listener |
| `studying_user_profile` | Academic context preference |
| `workout_user_profile` | Exercise/fitness context |
| `recommender` | Recommender instance with sample data |

## Test Results
```
============================== 43 passed in 0.06s ==============================
```

## Test Organization

Tests are organized into logical test classes:
- `TestSongDataclass`
- `TestUserProfileDataclass`
- `TestRecommenderCalculateScore`
- `TestContextualPreferences`
- `TestContextScore`
- `TestRecommenderRecommend`
- `TestExplainRecommendation`
- `TestValidateRecommendations`
- `TestGetRecommendationDocumentation`
- `TestLoadSongs`
- `TestRecommendSongs`
- `TestIntegration`

## Running Tests

```bash
# Run all tests with verbose output
python -m pytest tests/test_recommender.py -v

# Run specific test class
python -m pytest tests/test_recommender.py::TestContextualPreferences -v

# Run specific test
python -m pytest tests/test_recommender.py::TestContextualPreferences::test_studying_context_adjusts_targets -v
```

## Coverage Highlights

✓ All core Recommender methods covered
✓ Edge cases (empty inputs, boundary values)
✓ Context-aware logic (studying, workout, time-of-day)
✓ CSV loading and formatting
✓ Integration workflows
✓ Scoring algorithms and adjustments
✓ Validation and quality checks
✓ Documentation generation

## Notes

- Uses pytest fixtures for test data setup
- Includes temporary CSV file creation/cleanup for file loading tests
- Tests validate both correctness and format of outputs
- Context-based adjustments thoroughly tested across different scenarios
- Integration tests verify end-to-end workflows
