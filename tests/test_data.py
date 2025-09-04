from pathlib import Path

def test_processed_data_files_exist(config):

    processed_dir = Path(config["paths"]["processed_dir"])
    train_file = processed_dir / "train.csv"
    val_file = processed_dir / "val.csv"
    test_file = processed_dir / "test.csv"

    assert train_file.exists(), f"Train file {train_file} invalid or missing."
    assert val_file.exists(), f"Validation file {val_file} invalid or missing."
    assert test_file.exists(), f"Test file {test_file} invalid or missing."