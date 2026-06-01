from oracle.prompts import DirectLabelMapper


def test_label_normalization_and_resolution():
    mapper = DirectLabelMapper()
    assert mapper.resolve_label("  LATERAL   LYING ") == "Lateral lying"
    assert mapper.resolve_label("eating") == "eating"


def test_each_label_has_balanced_descriptions():
    mapper = DirectLabelMapper()
    for label in mapper.get_all_labels():
        assert len(mapper.get_positive_descriptions(label)) == 6
        assert len(mapper.get_negative_descriptions(label)) == 6
