# Pretrained detector models

Not committed. `make models` (or `riskfusion fetch-models`) downloads them and verifies SHA-256.

| File | Source | Licence |
|---|---|---|
| yolox_nano.onnx | Megvii YOLOX release 0.1.1rc0 | Apache-2.0 |
| det_500m.onnx (SCRFD) | InsightFace buffalo_s | weights: non-commercial research only |
| w600k_mbf.onnx (ArcFace) | InsightFace buffalo_s | weights: non-commercial research only |

The buffalo_s archive also contains a gender/age model. It is never extracted or loaded (SRS ETH-4).
