from unittest.mock import MagicMock, patch

from app.config.production_retrieval import FROZEN_EMBEDDING_MODEL


@patch("sentence_transformers.SentenceTransformer")
def test_prefetch_script_uses_frozen_model_by_default(mock_sentence_transformer: MagicMock):
    import importlib.util
    from pathlib import Path

    mock_sentence_transformer.return_value.get_sentence_embedding_dimension.return_value = 768

    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "prefetch_embedding_model",
        repo / "src" / "scripts" / "prefetch_embedding_model.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    mock_sentence_transformer.assert_called_once_with(FROZEN_EMBEDDING_MODEL)
