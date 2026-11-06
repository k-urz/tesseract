# Rotation constraint study

Config `66aebc68e650eb46`. xw perspective rotation.

Counts below are sampled INFINITESIMAL dimensions, excluding global rigid motion for the point model.

| Constraint model | Min | Max |
| --- | ---: | ---: |
| 32 lengths locked | 11 | 11 |
| 8 equal-length families | 18 | 18 |
| Families + y/z right angles | 15 | 15 |
| Lengths + selected angles coupled to phase | 1 | 1 |
| 4 aligned squares: conics + phase | 1 | 1 |

## Coupled geometry

Four aligned, expanding square frames have eight coordinates (X, B per frame).
Four conic constraints plus three source-phase equalities leave one local motion.
Positive phase dot products select the intended assembly branch.
The selected local complement contains 11 corner-angle constraints.
They are scalar target functions, NOT a count of physical hinges or actuators.

## Limits are insufficient

{'available': True, 'angle_deg': 0.0, 'shrink_factor': 0.999, 'all_length_limits_satisfied': True, 'max_corner_angle_change_deg': 0.0, 'equal_length_families_preserved': True, 'fixed_scale_path_constraint_error': 0.0050175300425561176, 'interpretation': 'Exact algebraic counterexample to limits/equalities selecting the fixed-scale path; not a hardware collision check.'}

## Limits of this result

- Infinitesimal point-model ranks are not finite mobility or stiffness certificates.
- Scheduled lengths/angles must be coupled to one common phase, not independently limited.
- Four-square construction assumes supported, aligned, square frames and positive phase-branch dot products.
- Cam profiles omit follower geometry, pressure angle, backlash, thickness and collision checks.
- The reduction is for xw rotation only; generic 4D rotations do not preserve these guides.
