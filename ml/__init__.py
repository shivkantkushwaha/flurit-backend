"""Flurit Machine Learning & Data Engineering Package.

This package encapsulates the end-to-end ML lifecycle for Flurit:
- Data validation and ingestion
- Time-aware customer snapshot generation
- Leakage-safe target labeling
- Feature engineering pipelines
- Chronological model evaluation
- Model artifact persistence & versioning
- SHAP explainability computation
- Unsupervised customer segmentation

BOUNDARY RULE:
Code in this package must remain independent of the FastAPI web serving layer.
Do not import FastAPI, Request, or HTTP routing primitives inside ml/.
"""
