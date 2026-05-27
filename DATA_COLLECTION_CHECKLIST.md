# Data Collection Checklist (Medicine -> Bowl)

Use this checklist when recording new VR teleop demos for `saivishwak/xlerobot-vr-teleop`.
Goal: improve grasp reliability and make the policy consistently pick medicine before placing into bowl.

## 1) Target Data Volume

- [ ] Collect **40-80 new successful episodes** (recommended total dataset: **55-95** episodes).
- [ ] Keep at least **80% clean successes** (avoid too many failed/noisy episodes).
- [ ] Record at existing settings (30 Hz, same joint order, same camera setup).

## 2) Scene + Robot Setup Consistency

- [ ] Use the same home pose used in training/inference.
- [ ] Keep camera mounts fixed (head, left wrist, right wrist) during the session.
- [ ] Keep table height and robot base position fixed.
- [ ] Use the same medicine object category/size as prior demos.

## 3) Episode Structure (Important)

Each episode should follow this timeline:

1. **Home settle** (about 2 seconds)
2. **Approach medicine**
3. **Grasp medicine**
4. **Lift and stabilize**
5. **Move to bowl**
6. **Place + release**
7. **Retreat**

Checklist:

- [ ] Start from home and hold still for ~2s before moving.
- [ ] Reach medicine first (do not move directly toward bowl first).
- [ ] Ensure clear, decisive gripper close around medicine.
- [ ] Lift after grasp (visible separation from table) before moving to bowl.
- [ ] Place into bowl and open gripper clearly.

## 4) Data Mix (High Impact)

Recommended split for new recordings:

- [ ] **20-30 episodes** focused on approach + grasp quality.
- [ ] **20-30 episodes** full medicine -> bowl trajectories.

Variation to include:

- [ ] Medicine orientation variations.
- [ ] Slight medicine position shifts.
- [ ] Mild bowl position shifts.
- [ ] Small lighting changes.

Avoid:

- [ ] Extreme scene changes outside expected deployment setup.
- [ ] Fast jerky teleop that creates inconsistent action labels.

## 5) Prompt/Task Labeling

Current full-task label is:
- `Pick up the medicine and place it in the bowl`

To strengthen phase behavior in future data:

- [ ] Consider collecting a subset with phase prompts:
  - `Pick up the medicine bottle`
  - `Place the medicine bottle in the bowl`
- [ ] Keep wording stable within each subset (avoid too many paraphrases).

## 6) Per-Episode Quality Gate

Before accepting an episode:

- [ ] Medicine is actually grasped (not just pushed/slid).
- [ ] Medicine is visibly lifted.
- [ ] Bowl placement is completed (release happens over bowl).
- [ ] No camera dropout/glitch during key grasp/place moments.
- [ ] No large accidental teleop spikes/collisions.

If an episode fails one or more gates:

- [ ] Re-record it (do not keep large numbers of low-quality attempts).

## 7) Session-End Sanity Checks

- [ ] Confirm episode count added and lengths look reasonable.
- [ ] Spot-check first/middle/last episodes visually for consistency.
- [ ] Verify task strings are correct and consistent.
- [ ] Run offline eval after upload/sync and compare against previous checkpoint behavior.

## 8) Retraining Guidance

- [ ] Retrain after each meaningful data batch (e.g., +20 episodes).
- [ ] Compare checkpoints (`005000`, `010000`, `015000`, `020000`, `last`) in offline eval.
- [ ] Keep best-performing checkpoint for on-robot tests.

---

If grasp remains unreliable after +40 clean episodes, prioritize additional grasp-centric episodes over adding more placement-only trajectories.



## Current Progress

| Label | Episodes |
|-------|----------|
| Pick up the medicine and place it in the bowl | 0-14 |
| Pick up the medicine | 15-34, 45-49 |
| Pick up the medicine and place in the bowl | 35-44 |