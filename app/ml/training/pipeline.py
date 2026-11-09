"""Model Pipeline and Preprocessing Factory.

Constructs training-only preprocessing transformers (ColumnTransformer)
and scikit-learn estimator pipelines for churn prediction.
"""

from typing import Any, Literal

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.features.schema import FEATURE_DEFINITIONS

CATEGORICAL_FEATURES = [f.name for f in FEATURE_DEFINITIONS if f.dtype == "category"]
NUMERICAL_FEATURES = [f.name for f in FEATURE_DEFINITIONS if f.dtype != "category"]


def build_preprocessor(scale_numericals: bool = False) -> ColumnTransformer:
    """Build ColumnTransformer for categorical and numerical features.

    Args:
        scale_numericals: If True, applies StandardScaler to numerical columns (e.g. for linear models).
                          If False, passes numericals through unscaled (e.g. for tree-based models).

    Returns:
        Unfitted ColumnTransformer.
    """
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    num_transformer = (
        StandardScaler() if scale_numericals else "passthrough"  # type: ignore[assignment]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ("num", num_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )


def create_model_pipeline(
    model_type: Literal[
        "logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting"
    ],
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Create full scikit-learn Pipeline combining preprocessor and classifier.

    Args:
        model_type: Model architecture to instantiate.
        random_state: Random seed for 100% deterministic reproducibility.
        **kwargs: Optional hyperparameter overrides passed to classifier.

    Returns:
        Unfitted Pipeline with steps ['preprocessor', 'classifier'].
    """
    if model_type == "logistic_regression":
        preprocessor = build_preprocessor(scale_numericals=True)
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            **kwargs,
        )

    elif model_type == "decision_tree":
        preprocessor = build_preprocessor(scale_numericals=False)
        classifier = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            **kwargs,
        )

    elif model_type == "random_forest":
        preprocessor = build_preprocessor(scale_numericals=False)
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    elif model_type == "hist_gradient_boosting":
        preprocessor = build_preprocessor(scale_numericals=False)
        classifier = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
