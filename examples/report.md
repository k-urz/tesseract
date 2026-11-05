# Sampled geometry study

Configuration: `66aebc68e650eb46`. Lengths: mm; angles: degrees.

| Scan | Samples | Edge range | Min clearance | Blocked | Two-stage edges fitting |
| --- | ---: | --- | ---: | ---: | ---: |
| prototype_full | 361 | 70.516–262.687 | -2.377 | 300 | 0/32 |
| prototype_candidate_coarse | 31 | 70.516–244.753 | -1.490 | 16 | 24/32 |
| prototype_candidate | 301 | 70.512–244.753 | -1.490 | 152 | 24/32 |
| target_full | 361 | 30.450–113.433 | -5.550 | 361 | 0/32 |
| target_candidate_coarse | 31 | 30.450–105.689 | -5.550 | 31 | 12/32 |
| target_candidate | 301 | 30.448–105.689 | -5.552 | 301 | 12/32 |

## Interpretation

- Negative clearance means overlapping conservative envelopes at a sampled pose.
- Rods sharing a hub are checked outside that hub, not skipped.
- Two-stage fit is a packaging/stroke check, not proof of joint travel or strength.
- A candidate can fail. Report failures before designing or ordering a complete model.
- Hardware dimensions stay fixed between the two model sizes; miniaturization does not assume infinitely small hardware.
- Sampling cannot certify absence of a collision between frames. No continuous-path, friction or holding claim is made.

## Coarse/fine comparison

```json
{
  "coarse_step_deg": 1.0,
  "fine_step_deg": 0.1,
  "max_edge_extremum_change_mm": 0.0037631593564100285,
  "min_clearance_change_mm": 0.0,
  "limitation": "Discrete samples only; convergence is not a continuous collision-free certificate.",
  "max_joint_extremum_change_deg": 0.015841111956028442
}
```
