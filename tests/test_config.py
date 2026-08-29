import pytest

from flowmia import FlowMIAConfig, load_config


def test_from_dict_to_dict_roundtrip():
    d = {
        "member_path": "m.csv",
        "non_member_path": "nm.csv",
        "synth_path": "s.csv",
        "test_path": "t.csv",
        "save_path": "out",
        "categorical_cols": ["proto"],
        "numerical_cols": ["td"],
        "ip_cols": ["srcip"],
        "label_col": "label",
        "use_wgan": True,
        "batch_size": 8,
        "num_epochs": 2,
        "fcheckpoint": 2,
    }
    config = FlowMIAConfig.from_dict(d)
    assert config.to_dict() == d


def test_missing_required_field_raises():
    with pytest.raises(TypeError):
        FlowMIAConfig.from_dict({"member_path": "m.csv"})


def test_empty_required_field_raises():
    with pytest.raises(ValueError):
        FlowMIAConfig(
            member_path="",
            non_member_path="nm.csv",
            synth_path="s.csv",
            test_path="t.csv",
            save_path="out",
        )


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        FlowMIAConfig.from_dict({
            "member_path": "m.csv",
            "non_member_path": "nm.csv",
            "synth_path": "s.csv",
            "test_path": "t.csv",
            "save_path": "out",
            "not_a_real_key": 1,
        })


def test_load_config_reads_yaml_example():
    config = load_config("examples/configs/netshare_example.yaml")
    assert config.member_path == "datasets/real/cidds_train.csv"
    assert config.categorical_cols == ["proto"]
    assert config.use_wgan is True
