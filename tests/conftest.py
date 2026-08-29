import numpy as np
import pandas as pd
import pytest


def _make_flow_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "srcip": rng.integers(0, 2**32 - 1, n),
        "dstip": rng.integers(0, 2**32 - 1, n),
        "srcport": rng.integers(0, 65535, n),
        "dstport": rng.integers(0, 65535, n),
        "td": rng.random(n) * 100,
        "pkt": rng.integers(1, 1000, n),
        "byt": rng.integers(1, 100000, n),
        "proto": rng.choice(["TCP", "UDP", "ICMP"], n),
        "label": rng.choice([0, 1], n),
    })


@pytest.fixture
def tiny_flow_config(tmp_path) -> dict:
    """Small, fully synthetic FlowMIA config dict pointing at CSVs in tmp_path."""
    member = _make_flow_df(40, seed=1)
    non_member = _make_flow_df(40, seed=2)
    synth = _make_flow_df(40, seed=3)
    test = _make_flow_df(20, seed=4)

    member_path = tmp_path / "member.csv"
    non_member_path = tmp_path / "non_member.csv"
    synth_path = tmp_path / "synth.csv"
    test_path = tmp_path / "test.csv"
    member.to_csv(member_path, index=False)
    non_member.to_csv(non_member_path, index=False)
    synth.to_csv(synth_path, index=False)
    test.to_csv(test_path, index=False)

    return {
        "member_path": str(member_path),
        "non_member_path": str(non_member_path),
        "synth_path": str(synth_path),
        "test_path": str(test_path),
        "save_path": str(tmp_path / "out"),
        "categorical_cols": ["proto"],
        "numerical_cols": ["srcport", "dstport", "td", "pkt", "byt"],
        "ip_cols": ["srcip", "dstip"],
        "label_col": "label",
        "use_wgan": True,
        "batch_size": 8,
        "num_epochs": 2,
        "fcheckpoint": 2,
    }
