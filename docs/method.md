# Method note: Differential game to Deep RL solver

## Mapping
We discretize the continuous-time differential game into a Markov game:
- State: `s_t = [K_t, t/H, mode_id, task_id, obs(phi_t)]`
- Action:
  - Nash: `(E_T, E_O)`
  - Stackelberg: `(E_T, E_O, theta)` with bilevel semantics
  - Cooperative: joint `(E_T, E_O)`
- Transition follows Euler-Maruyama discretization with stochastic noise.

## Solvers
1. **Nash solver (MADDPG-style)**: two deterministic actors + centralized critics. We choose MADDPG-style over MASAC to keep implementation lightweight and reproducible under minimal dependencies.
2. **Stackelberg solver**: bilevel actor-critic with alternating updates:
   - follower best-response learner `pi_O(E_O|s,E_T,theta)`
   - leader learner `pi_T(E_T,theta|s)`
3. **Cooperative solver**: single joint actor-critic; supports feedforward or GRU actor.
4. **Robust extension**:
   - domain randomization via episode/step parameter sampling
   - risk-sensitive penalty (`mean - lambda * variance`) proxy in cooperative update.

## Limitations and failure modes
- Deterministic-policy critics are sensitive to reward scale and action saturation.
- Stackelberg alternating interval is hyperparameter-sensitive; too frequent leader updates can destabilize follower BR learning.
- Hidden-parameter mode with short horizon may not fully reveal latent dynamics; recurrent policies can still underfit.
- Results are intended as **recovery/robustness approximations**, not proofs replacing analytic equilibria.
