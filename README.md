# Imitation Learning with Latent Market Parameters in Continuous-Time Portfolio Control

Benchmarks imitation learning against optimal stochastic control solutions for
portfolio optimization, and uses the learned policy to improve PPO training.

- We compare Behavior Cloning (BC) and DAgger against the closed-form optimal
  policies of the Merton model and a jump-diffusion extension, under a
  trajectory-dependent setting where the market parameters (mu, sigma) are drawn
  randomly per trajectory and unobserved by the agent.
- We show that a feed-forward network augmented with running sample statistics
  partially tracks the optimal policy, but remains too volatile; an RNN-based
  BC policy that reads the full trajectory history implicitly infers the latent
  parameters and outperforms the naive Merton plug-in estimator in
  out-of-sample terminal-wealth stability and CRRA utility.
- We show that DAgger substantially reduces both policy error and error
  accumulation over time compared to BC in the jump-diffusion setting, where
  even recovering a constant policy is non-trivial.
- We show that hot-starting PPO with the IL-learned network stabilizes training
  and achieves higher median terminal wealth compared to randomly initialized
  PPO, confirming the value of transfer learning in this setting.

Presented at the 2026 SSC Student Conference (poster) and the 2026 SSC Annual Conference (oral).

## Setting up the project

```bash
pip install -r requirements.txt
```

## Reproducing results
Simply run `run_il.py` and `run_ppo.py` to reproduce the results. The saved plots are in `/plots`.
```bash
python [run_il.py|run_ppo.py]
```
