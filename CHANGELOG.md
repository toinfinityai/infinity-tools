# Changelog
All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.5.0] - 2022-08-16

### Added
- Added `spills` submodule to support interactions with the Spills API and related visualizations.

## [0.4.0] - 2022-08-05

### Changed

- Remove hardcoded parameter definitions/constraints and instead retrieve dynamically from the API.
- Update visualization tooling for job parameter distributions.

## [0.3.1] - 2022-07-04

### Changed

- VisionFit bumped to `v0.3.1`.

### Fixed

- Use the correct image resolution in exported camera matrices for VisionFit.

## [0.3.0] - 2022-06-13

### Added
- Support for up to 20 repetitions in both SenseFit and VisionFit.
- Added many more exercises to SenseFit and VisionFit generators.
- New VisionFit parameters:
  - `rel_baseline_speed`
  - `camera_distance`
  - `avatar_identity`
  - `relative_height`
  - `relative_weight`
- New SenseFit parameters:
  - `rel_baseline_speed`
  - `wrist_offset_deg`

### Changed
- Polling mechanism used by `await_jobs` and `get_completed_jobs` now takes advantage of new backend updates to be more efficient and use only a single HTTP request, regardless of batch size.
- Fixed/updated documentation for various parameters and their constraints.
- VisionFit bumped to `v0.3.0`.
- SenseFit bumped to `v0.2.0`.
- Maximum allowable value for `kinematic_noise_factor` increased to 2.0 for SenseFit.
- Renamed existing exercises to follow a new naming convention.
- Increased the default kinematic variation for select exercises: `PUSHUP`, `BURPEE`, and `PUSHUP-EXPLOSIVE`.

### Removed
- `seconds_per_rep` parameter removed from both VisionFit and SenseFit.

## [0.2.0] - 2022-05-25

### Added

- Added Github Action for Python `black` code formatting.
- Added 30 frames-per-second option to `frame_rate` parameter for VisionFit.
- New vertex keypoints added to the output of VisionFit.

### Changed

- Whitespace changes from applying `black` formatter.
- VisionFit bumped to `v0.2.0`.
- SenseFit bumped to `v0.1.1`.
- Serial request delay changed to parameter from hardcoded value in `await_jobs`.
- Removed `assert`s and replaced with raised `Exception`s throughout codebase.

### Fixed

- Removed broken "DUMBBELL_DEADLIFT" exercise for SenseFit.
