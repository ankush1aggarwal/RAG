from app.config.settings import Settings


def test_resolved_fiqa_corpus_limit_full_when_zero():
    settings = Settings(_env_file=None, fiqa_corpus_limit=0)
    assert settings.resolved_fiqa_corpus_limit() is None


def test_resolved_fiqa_corpus_limit_positive():
    settings = Settings(_env_file=None, fiqa_corpus_limit=1000)
    assert settings.resolved_fiqa_corpus_limit() == 1000
