from georeset_corpus_assessment import cli, engine, settings


def test_settings_pin_the_model_and_dataset() -> None:
    assert settings.MODEL_ID == "LiquidAI/LFM3-1.7B"
    assert len(settings.MODEL_REVISION) == 40
    assert len(settings.SOURCE_DATASET_REVISION) == 40
    assert settings.RESULTS_DATASET == "NoeFlandre/GeoReSeT-Corpus-Assessment"


def test_adapters_import_without_gpu_dependencies() -> None:
    assert callable(cli.main)
    assert engine.TransformersEngine.__name__ == "TransformersEngine"


def test_finish_reason_is_stop_on_eos_and_length_at_the_limit() -> None:
    assert engine._finish_reason([5, 2], limit=10, eos_token_id=2) == "stop"
    assert engine._finish_reason([5, 6], limit=2, eos_token_id=2) == "length"
    assert engine._finish_reason([5, 6], limit=10, eos_token_id=None) == "stop"
