# Changelog

## 0.2.2 - 2026-10-23

### Fixed

- Allow pip to select a Flower and Ray combination compatible with the user's
  supported Python version and platform.
- Remove the direct `ray==2.51.1` pin and rely on the Ray version declared by
  the `flwr[simulation]` dependency.
- Remove the artificial Flower `<1.31.0` upper bound while retaining
  `flwr[simulation]>=1.30.0` as the minimum supported version.
- Keep the Flower compatibility test on the public `flwr.common.ConfigRecord`
  import path, which works with the validated Flower 1.30 and 1.42 releases.

No federated-learning algorithm, trainer, adversarial attack, metric, or public
FAL API behavior changed in this release.
