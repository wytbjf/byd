# Unified notation and parameter conventions

- Time is discretized with step `dt`; horizon length is `H`.
- State uses `K_t` (AI diffusion effort stock), `mode_id`, `task_id`, and optional parameter exposure vector.
- We unify TEU direction as **higher TEU means stronger AI-enabled productivity and lower effective disutility coefficient** through factors `(1 - epsilon * a_i * TEU_i)`.
- `task_type=routine` emphasizes higher `alpha` and lower `beta`; `task_type=creative` allows higher `beta` and lower decay `delta`.
- `K0_high > K0_low`; initial `K_0 ~ U[K0_low, K0_high]`.
- Cost-share `theta` only exists in Stackelberg and is constrained by sigmoid to `[0,1]`.
- Efforts `E_T, E_O` are constrained non-negative via softplus action decoding.
- Feasibility constraints are enforced in parameter sampling:
  - `1 - epsilon * a1 * TEU1 > 0`
  - `1 - epsilon * a2 * TEU2 > 0`
  - Optional analytic regime switch: `2 * omega1 > omega2`.
- Hidden-parameter POMDP mode masks `phi` from observation, requiring history/recurrent policy for better performance.
