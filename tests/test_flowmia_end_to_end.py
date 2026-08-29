from sklearn.tree import DecisionTreeClassifier

from flowmia import FlowMIA


def test_flowmia_builds_and_transforms(tiny_flow_config):
    flowmia_instance = FlowMIA(config=tiny_flow_config)

    assert flowmia_instance.X_member.shape[0] == 40
    assert flowmia_instance.X_non_member.shape[0] == 40
    assert flowmia_instance.X_synth.shape[0] == 40
    assert flowmia_instance.X_member.shape == flowmia_instance.X_synth.shape


def test_evaluate_privacy_dcr_only(tiny_flow_config):
    flowmia_instance = FlowMIA(config=tiny_flow_config)
    results = flowmia_instance.evaluate_privacy(attacks=["dcr"], test_size=10)

    assert 0.0 <= results["dcr"].auc <= 1.0


def test_evaluate_fidelity(tiny_flow_config):
    flowmia_instance = FlowMIA(config=tiny_flow_config)
    fidelity = flowmia_instance.evaluate_fidelity()

    assert set(fidelity) == set(
        tiny_flow_config["categorical_cols"]
        + tiny_flow_config["numerical_cols"]
        + tiny_flow_config["ip_cols"]
        + [tiny_flow_config["label_col"]]
    )


def test_evaluate_utility(tiny_flow_config):
    flowmia_instance = FlowMIA(config=tiny_flow_config)
    utility = flowmia_instance.evaluate_utility([DecisionTreeClassifier()])

    assert "DecisionTreeClassifier" in utility
    assert "RTR" in utility["DecisionTreeClassifier"]
    assert "TSTR" in utility["DecisionTreeClassifier"]


def test_flowmiagan_end_to_end(tiny_flow_config):
    flowmia_instance = FlowMIA(config=tiny_flow_config)
    results = flowmia_instance.flowmiagan(test_size=10)

    assert 0.0 <= results["auc"] <= 1.0
