def test_exiffi_uses_exiffi_path_weight():
    import numpy as np
    from sklearn.ensemble import IsolationForest

    from efd.explanations.exiffi import ExIFFI

    rng = np.random.RandomState(42)

    X = rng.normal(size=(100, 4))

    model = IsolationForest(
        n_estimators=1,
        max_samples=32,
        random_state=42,
    )

    model.fit(X)

    feature_names = [
        "f0",
        "f1",
        "f2",
        "f3",
    ]

    explainer = ExIFFI(
        model=model,
        feature_names=feature_names,
    )

    x = X[0]

    explanation = explainer.explain(x)

    tree = model.estimators_[0].tree_

    indicator = model.estimators_[0].decision_path(
        x.reshape(1, -1)
    )

    path_nodes = indicator.indices[
        indicator.indptr[0]:
        indicator.indptr[1]
    ]

    expected = np.zeros(
        X.shape[1],
        dtype=float,
    )

    for node_id in path_nodes:

        feature = tree.feature[node_id]

        if feature < 0:
            continue

        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if x[feature] <= tree.threshold[node_id]:
            child = left
        else:
            child = right

        n_parent = tree.n_node_samples[node_id]
        n_child = tree.n_node_samples[child]

        expected[feature] += (
            n_parent / n_child
        )

    actual = explanation.contributions

    assert np.allclose(
        actual,
        expected,
    )