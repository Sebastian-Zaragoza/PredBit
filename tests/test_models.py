from pathlib import Path

def test_model_file_exists(config):
    model_dir = Path(config["paths"]["model_dir"])
    model_file = model_dir / "random_forest_pipeline.pkl"
    assert model_file.exists(), f"Model file {model_file} invalid or missing."

def test_model_can_predict(model):
    sample_data = {
        'daily_screen_time_min': [150],
        'notification_count': [120],
        'social_media_time_min': [90],
        'mood_score': [7],
    }

    prediction = model.predict(sample_data)
    assert len(prediction) == 1, "The prediction failed."