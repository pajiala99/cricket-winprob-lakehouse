from cricklake.config import PROJECT_ROOT, load_config


def test_real_profile_overrides_paths_and_inherits_the_rest():
    real = load_config(PROJECT_ROOT / "conf" / "real.yaml")
    base = load_config(PROJECT_ROOT / "conf" / "pipeline.yaml")
    assert "extends" not in real
    assert real["paths"]["lakehouse"] == "data/real/lakehouse"
    assert real["paths"]["lakehouse"] != base["paths"]["lakehouse"]
    assert real["model"] == base["model"] and real["spark"] == base["spark"]
    assert real["source"]["base_url"] == base["source"]["base_url"]  # nested dicts merge key by key
    assert "ipl_json" in real["source"]["competitions"]
