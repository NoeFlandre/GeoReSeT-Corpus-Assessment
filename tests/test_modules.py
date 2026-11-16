from georeset_corpus_assessment import cli, engine, settings


def test_settings_pin_the_model_and_dataset() -> None:
    assert settings.MODEL_ID == "LiquidAI/LFM3-1.7B"
    assert len(settings.MODEL_REVISION) == 40
    assert len(settings.SOURCE_DATASET_REVISION) == 40
    assert settings.RESULTS_DATASET == "NoeFlandre/GeoReSeT-Corpus-Assessment"


def test_adapters_import_without_gpu_dependencies() -> None:
    assert callable(cli.main)
    assert engine.VllmEngine.__name__ == "VllmEngine"
