import os
from unittest.mock import MagicMock, patch


def test_kaggle_uses_cached_file_without_download(tmp_path):
    """Если файл уже есть — KaggleApi не должен создаваться вообще."""
    from src.core.data.fetcher import RawDataFetcher

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text,label\nhello,0\nworld,1")

    fetcher = RawDataFetcher(
        source_type="kaggle",
        raw_dir=str(tmp_path),
        dataset_name="user/dataset",
        file_name="data.csv",
    )

    # Патчим там где fetcher делает import — внутри _load_kaggle
    with patch("src.core.data.fetcher.KaggleApi") as mock_api_class:
        with patch("src.core.data.fetcher.load_dataset") as mock_load:
            mock_load.return_value = MagicMock()
            fetcher.load()
            mock_api_class.assert_not_called()

    mock_api_instance = MagicMock()

    with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "testkey"}):
        with patch("src.core.data.fetcher.KaggleApi", return_value=mock_api_instance):
            with patch("src.core.data.fetcher.load_dataset") as mock_load:
                mock_load.return_value = MagicMock()
                fetcher.load()
                mock_api_instance.authenticate.assert_called_once()
                mock_api_instance.dataset_download_files.assert_called_once_with(
                    "user/dataset", path=str(tmp_path), unzip=True
                )
