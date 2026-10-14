# Fall Detection Prototype: Problems and Development Plan

Reviewed on 14 October 2026. Repository: https://github.com/anonnaspondon/FallDetection

Source baseline: commit `09431b6125a827789e9fdf904f6c4632c42442e7`.

This is a source-code review, not a runtime benchmark or accuracy evaluation. Findings describe the inspected implementation; they do not establish that every failure occurs in every deployment.

## Intended application

Detect human falls from a camera near ground level, representing a rover-mounted viewpoint. The initial prototype may keep the camera separate from the rover, but must evaluate it at the intended mounting height and angle. Develop on a desktop, then verify inference on a Windows laptop with an Intel i5-8250U, 8 GB RAM, and NVIDIA MX150. CPU inference is the initial portable baseline.

The completed system should provide processed video, possible-fall alerts, and manual ESP32 rover controls through one dashboard. Remote access may use Cloudflare Tunnel. An alert must not automatically move the rover.

## Existing implementation

- `FallDetection/server/detector_rule.py`: YOLO pose estimation, overlap-based tracking, feature buffers, and rule-based alerts.
- `FallDetection/server/features.py`: posture/motion features and fall rules.
- `FallDetection/server/live_view.py`: local webcam diagnostic display.
- `FallDetection/server/server.py`: FastAPI JPEG-frame receiver and health/debug endpoints.
- `FallDetection/server/webcam_sender.py`: webcam JPEG sender.
- `FallDetection/server/yolov8n-pose.pt`: pose-estimation weights, not a trained fall-classification model.

The detector comments mention a future GRU, but no GRU training pipeline or trained fall-classifier weights appear in the reviewed tree.

## 1. Lying posture can be mistaken for an observed fall

**Evidence:** `rule_based_fall()` returns true when either `lying` or `falling` is true. The lying condition does not require a preceding descent or upright-to-low transition.

**Impact:** Intentional lying down or an already-horizontal person can trigger an alert. Post-fall posture and an observed fall event are different observations.

**Required work:** Track prior posture, descent, transition to low posture, and persistence. Keep "person lying down" separate from "possible fall observed" when evidence supports only the former.

**Verification:** Include intentional lying down, horizontal startup posture, brief bending, sitting rapidly, and labelled falls in held-out tests.

## 2. Downward movement is measured in a moving coordinate system

**Evidence:** `extract_features()` computes vertical position as `(center_y - box_top) / box_height`, then differences that value over time.

**Impact:** A person and bounding box moving downward together can retain the same relative position, hiding descent. Box shape changes can also create apparent movement.

**Required work:** Measure movement in consistent image coordinates, normalize by an appropriate person scale, and explicitly handle scale changes. Validate the feature against known movement trajectories. Camera-motion compensation or detection while parked will be needed for a moving-rover deployment.

## 3. Alarm timing depends on processing frame rate

**Evidence:** Five qualifying frames trigger an alarm and 30 normal frames clear it. The feature buffer has 30 frames, and velocity uses server processing time.

**Impact:** The same video can produce different timing at different inference rates or network delays. `fall_frames` decays by one on negative evidence rather than requiring uninterrupted positive evidence.

**Required work:** Use monotonic capture/media timestamps, time-based evidence windows, and durations in seconds. Specify whether persistence must be continuous or accumulated. Recorded-video evaluation must use media time rather than replay speed.

**Verification:** Replay identical recordings at different processing speeds and sampling rates; compare classifications and alert timing.

## 4. Critical missing landmarks are insufficiently guarded

**Evidence:** Posture features use shoulders, hips, ankles, and the nose, but the rule gate checks mean confidence across keypoints rather than confidence for each required landmark.

**Impact:** Good confidence on unrelated landmarks can hide unreliable points used in an angle or movement feature. Missing coordinates can create misleading measurements.

**Required work:** Validate each landmark needed for a feature, reject invalid/off-image coordinates, and introduce an `UNKNOWN` tracking state. If partial-body inference is supported later, define and evaluate that behavior separately.

## 5. Tracking and tracking-loss recovery need strengthening

**Evidence:** `RuleFallDetector` matches boxes using IoU above 0.35 and expires tracks after 1.5 seconds. The live viewer selects the highest-confidence person each frame and retains one feature history.

**Impact:** Rapid posture changes can break identity continuity. The viewer can combine observations from different people or across disappearance/reappearance.

**Required work:** Return explicit detection-to-track assignments, maintain stable identities, and reset or invalidate temporal evidence after significant tracking loss or identity changes. Treat occlusion as uncertainty.

**Verification:** Test rapid falls, short occlusions, people entering/leaving, and multiple-person crossings.

## 6. Ground-level performance is unproven

**Evidence:** No ground-level evaluation report, labelled test set, or viewpoint-specific accuracy results appear in the reviewed repository.

**Impact:** Low camera angles alter apparent body orientation and visibility. A detector working from desk height may fail from rover height.

**Required work:** Record mounting height, tilt, subject distance, floor coverage, and lighting. Evaluate low-angle falls and ordinary activities. Include representative low-angle data in development/training and reserve independent sessions or subjects for final testing.

Stationary low-camera tests do not establish reliability while driving. Start with detection while parked and evaluate motion-induced false alarms separately.

## 7. No trained temporal fall classifier is supplied

**Evidence:** YOLO estimates pose; handwritten rules make the fall decision. A feature buffer and a comment about a future GRU do not constitute a trained classifier.

**Required work:** First improve and evaluate the rule baseline. If reliable pose features are insufficient for the decision, train a small temporal classifier on labelled landmark sequences and compare it against the rules.

Do not train the pose network from scratch by default. Compare pretrained pose models first. A fall classifier cannot reliably compensate for consistently missing body points.

## 8. Inference blocks the asynchronous request handler

**Evidence:** The asynchronous `/frame` route calls synchronous `detector.process_frame()` directly.

**Impact:** Inference occupies the event loop and can delay health requests and future control traffic. Moving inference to workers also requires deliberate access control for shared detector state.

**Required work:** Separate capture/inference from web and rover-control handling. Use bounded buffers, discard stale live frames, preserve timestamps, and ensure ordered processing for each camera. Define single-camera behavior or isolate state per camera/device before accepting multiple sources.

**Verification:** Under inference load, measure status responsiveness and STOP/command-expiry behavior.

## 9. The shared dashboard and rover integration are missing

**Evidence:** The repository provides a local viewer and JSON API, but no shared browser dashboard, video-output endpoint, ESP32 relay, or remote-control session handling.

**Required work:** Add annotated video, timestamped alerts, acknowledgement, camera/model/rover status, press-and-hold driving controls, and STOP. Browser commands should go through the application to the existing ESP32 protocol, not directly to a private device address.

Use one active control owner, allowlisted commands, fresh-input expiry, stopped reconnections, and authentication before remote exposure. A relay must never repeat a stale movement command indefinitely. HTTPS deployment needs secure WebSockets and same-application video/status routes.

## 10. Evaluation and laptop deployment are not demonstrated

**Evidence:** No automated tests or accuracy report appear in the reviewed tree. Dependency pins are present, but their compatibility with the target laptop has not been tested in this review.

**Required work:** Provide a verified Windows environment, model provenance, live-webcam and recorded-video modes, and repeatable logs. Measure event-level recall/precision, false alarms per hour, alert delay, processing speed, frame age, and process RAM. Define event matching and target performance before final evaluation.

Keep training/calibration recordings separate from test recordings; do not split adjacent frames from a single clip across both. Public datasets should be matched to the deployment's camera modality and viewpoint and supplemented with representative recordings.

Add meaningful tests for timestamp handling, tracking loss, uncertain landmarks, alert/reset behavior, and ordinary activities. Test camera loss and overload. Distinguish rule scores from calibrated probabilities: the current fixed values 0.7/0.9 are heuristic scores, not measured confidence.

## 11. Reuse and distribution terms need clarification

**Evidence:** No repository license file was present and GitHub metadata reported no license.

**Required work:** Ask the author to clarify code reuse terms before incorporating the implementation into another project. Record attribution and review the separate upstream code, pretrained-model, and dataset terms. Public visibility alone does not provide an explicit reuse license.

## Recommended implementation order

1. Confirm reuse terms and create a reproducible CPU installation.
2. Compare YOLOv8n-pose and MediaPipe on the same rover-height footage and laptop.
3. Fix coordinate-based motion features, per-landmark confidence checks, and timestamp handling.
4. Improve tracking and a temporal rule baseline with explicit uncertainty.
5. Evaluate on representative labelled low-angle recordings.
6. Train a small temporal classifier only if the baseline results justify it.
7. Integrate the dashboard and rover relay, then add protected Cloudflare access.
8. Complete held-out accuracy, performance, and failure-case testing on the demonstration laptop.

## Acceptance criteria to agree before claiming success

- Full-body tracking is sufficiently reliable from the specified mounting geometry.
- Agreed event recall, false-alarm, and delay targets are met on held-out recordings.
- The complete application runs on the laptop without unbounded frame delay.
- Tracking loss and camera loss visibly produce uncertainty/unavailability.
- Rover stopping remains responsive during overload and communication loss.

No source-only review can guarantee successful detection of every fall. This document is a development checklist, not an accuracy certification.
