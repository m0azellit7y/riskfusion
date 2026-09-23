# MediaPipe models (converted to ONNX)

Committed with the project so no extra download is needed. Source: the `.tflite` models bundled in the
`mediapipe==0.10.14` wheel from PyPI, converted with `tf2onnx 1.16.1` (opset 13) so they run on ONNX Runtime
(the same engine as the other detectors), on Windows, Linux and macOS, without MediaPipe's own dependencies.

| File | Original | Used for | Licence |
|---|---|---|---|
| face_landmark.onnx | face_landmark.tflite (468 points) | eye openness (blinks) and micro-motion -> liveness | Apache-2.0 |
| pose_landmark_full.onnx | pose_landmark_full.tflite (33+6 points) | shoulders, elbows, wrists -> body pose, reach out of frame | Apache-2.0 |
| palm_detection_lite.onnx | palm_detection_lite.tflite | number of hands in view | Apache-2.0 |

Copyright Google LLC, licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
The MediaPipe person detector could not be converted and is not needed: the person region is taken from the
SCRFD face detection.
