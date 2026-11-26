"""PPO for the robotaxi environment (IAEML, Assignment 2).

Everything that touches the environment is compiled into one XLA program:
the rollout (``lax.scan`` over time, ``vmap`` over parallel envs), GAE and
the PPO epochs. Python only calls ``update`` in a loop and logs.

Run from the repository root:

    # smoke test (a few minutes): checks that everything compiles and learns
    python src/algorithms/train_ppo.py --map 0 --num-envs 64 --total-timesteps 300000

    # full run
    python src/algorithms/train_ppo.py --map 0 --num-envs 256 --total-timesteps 10000000

Outputs (in ``--out-dir``, default ``<repo>/checkpoints``):
    ppo_map{N}_best.msgpack / ppo_map{N}_last.msgpack   policy + obs statistics
    ppo_map{N}.json                                      config used for eval
    ppo_map{N}_log.csv                                   training curves
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, NamedTuple

import equinox
import flax.linen as nn
import jax
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes, to_bytes
from flax.training.train_state import TrainState
from jax import numpy as jnp

# make `env`, `utils`, ... (src/) importable no matter where the script is run from
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from env.robotaxi import RoboTaxiEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Environment factory (shared with eval_ppo.py)
# ---------------------------------------------------------------------------


def make_env(map_id, max_steps, action_repeat, replan_period, path_length,
             num_rays=32, perception_radius=5.0, progress_on_path=False,
             plan_static_only=False, plug_slits=False):
    """Create (env_params, init_state) for the training configuration."""
    return RoboTaxiEnv.init_params(
        key=jax.random.PRNGKey(0),
        map_id=map_id,
        max_steps=max_steps,
        path_length=path_length,
        perception_radius=perception_radius,
        num_ray_sensors=num_rays,
        overrides=dict(action_repeat=action_repeat, replan_period=replan_period,
                       progress_on_path=progress_on_path,
                       plan_static_only=plan_static_only, plug_slits=plug_slits),
    )


# ---------------------------------------------------------------------------
# Actor-critic network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Two separate MLPs (actor, critic); Gaussian policy with a
    state-independent log-std."""

    hidden: int = 128
    act_dim: int = 2
    init_log_std: float = -0.7

    @nn.compact
    def __call__(self, x):
        a = x
        for _ in range(2):
            a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                                 bias_init=constant(0.0))(a))
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(a)
        log_std = self.param("log_std", constant(self.init_log_std), (self.act_dim,))

        c = x
        for _ in range(2):
            c = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                                 bias_init=constant(0.0))(c))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return mean, log_std, jnp.squeeze(value, -1)


def gaussian_logprob(mean, log_std, action):
    z = (action - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


# ---------------------------------------------------------------------------
# Running observation statistics
# ---------------------------------------------------------------------------


class RunningStats(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def stats_init(dim):
    return RunningStats(jnp.zeros(dim), jnp.ones(dim), jnp.asarray(1e-4))


def stats_update(s: RunningStats, batch):
    """Merge the statistics of ``batch`` (n, dim) into ``s`` (parallel Welford)."""
    n = batch.shape[0]
    b_mean, b_var = batch.mean(0), batch.var(0)
    delta = b_mean - s.mean
    total = s.count + n
    new_mean = s.mean + delta * n / total
    m2 = s.var * s.count + b_var * n + delta ** 2 * s.count * n / total
    return RunningStats(new_mean, m2 / total, total)


def normalize(s: RunningStats, x):
    return jnp.clip((x - s.mean) / jnp.sqrt(s.var + 1e-8), -10.0, 10.0)


# ---------------------------------------------------------------------------
# Rollout containers
# ---------------------------------------------------------------------------


class Carry(NamedTuple):
    states: Any            # batched RoboTaxiState
    obs: jnp.ndarray       # (N, D) raw (unnormalized) observation vectors
    ep_ret: jnp.ndarray    # (N,) return of the running episode
    ep_len: jnp.ndarray    # (N,) length of the running episode
    rng: jnp.ndarray
    t: jnp.ndarray         # global env-step counter (drives the path replanning)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class Batch(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    adv: jnp.ndarray
    target: jnp.ndarray


# ---------------------------------------------------------------------------
# Random starts (exploration) and evaluation from the true start
# ---------------------------------------------------------------------------


def make_spawn_pool(env_params, init_state, num, seed, t_max=20.0):
    """A pool of ``num`` initial states: random free position and heading of the
    car, random phase of the moving obstacles, DP path computed for that state.

    Why: from the fixed start the agent never gets as far as the goal, so it
    never sees the terminal bonus and learns to wait in front of the hardest
    passage. Random starts let it experience the whole route.
    """
    rng = np.random.default_rng(seed)
    static = np.asarray(init_state.static_obstacles, dtype=np.float32)
    kin0 = np.asarray(init_state.kinematic_obstacles, dtype=np.float32)
    vel = np.asarray(init_state.kinematic_obst_velocities, dtype=np.float32)
    goal = np.asarray(init_state.goal_pos, dtype=np.float32)
    h, w = env_params.map_height_width
    dims = np.array([w, h], dtype=np.float32)  # world x in [0, w), y in [0, h)

    # candidate cells: not inside a wall (touching is allowed, like at the real
    # start) and not on the goal
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    cells = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    d_static = np.linalg.norm(cells[:, None, :] - static[None, :, :], axis=-1).min(axis=1)
    cells = cells[d_static >= 1.0 - 1e-6]
    cells = cells[np.linalg.norm(cells - goal, axis=1) >= 1.5]

    pos = np.zeros((num, 2), np.float32)
    kin = np.zeros((num,) + kin0.shape, np.float32)
    for i in range(num):
        for _ in range(100):
            k = np.mod(kin0 + vel * rng.uniform(0.0, t_max), dims)
            c = cells[rng.integers(len(cells))]
            if len(k) == 0 or np.linalg.norm(k - c, axis=1).min() >= 2.0:
                break  # no moving obstacle too close to the spawn point
        else:  # fallback: the default start
            c, k = np.asarray(init_state.agent_pos, np.float32), kin0
        pos[i], kin[i] = c, k
    heading = rng.uniform(-np.pi, np.pi, num).astype(np.float32)
    fwd = np.stack([np.cos(heading), np.sin(heading)], axis=1).astype(np.float32)

    pool = jax.tree.map(lambda x: jnp.broadcast_to(x, (num,) + jnp.shape(x)), init_state)
    pool = equinox.tree_at(
        lambda st: (st.agent_pos, st.heading, st.agent_forward_dir, st.kinematic_obstacles),
        pool,
        (jnp.asarray(pos), jnp.asarray(heading), jnp.asarray(fwd), jnp.asarray(kin)),
    )
    # paths (and the progress potential) for every spawn state
    return jax.jit(jax.vmap(lambda st: RoboTaxiEnv.replan(st, env_params)))(pool)


def make_true_start_eval(net, env_params, init_state):
    """Deterministic episode from the real start; returns a stats dict.
    This is the metric that matters, the training stats come from random starts."""
    K = env_params.replan_period
    key0 = jax.random.PRNGKey(0)

    @jax.jit
    def act(params, stats, vec):
        mean, _, _ = net.apply(params, normalize(stats, vec))
        return jnp.clip(mean, -1.0, 1.0)

    def run(params, stats):
        obs, state = RoboTaxiEnv.reset(key0, env_params, init_state)
        ret, done, t, info = 0.0, False, 0, {}
        while not done:
            action = act(params, stats, obs.as_vector())
            obs, state, reward, done, info = RoboTaxiEnv.step(key0, state, action, env_params)
            t += 1
            if K > 0 and t % K == 0:
                state = RoboTaxiEnv.replan(state, env_params)
                obs = RoboTaxiEnv.get_observation(state, env_params)
            ret += float(reward)
            done = bool(done)
        return dict(success=bool(info["success"]), crash=bool(info["collision"]),
                    d_goal=float(info["distance_to_goal"]), ret=ret, steps=t)

    return run


# ---------------------------------------------------------------------------
# Build the compiled training step
# ---------------------------------------------------------------------------


def build(env_params, init_state, net, cfg):
    """Return (init_carry, update). ``update`` is a jitted function that
    collects a rollout and performs the PPO epochs."""
    N, T = cfg.num_envs, cfg.rollout_len
    B = N * T
    assert B % cfg.num_minibatches == 0, "num_envs * rollout_len must divide num_minibatches"
    K = env_params.replan_period
    gamma, lam = cfg.gamma, cfg.gae_lambda

    # a batch of N copies of the initial state (used for autoreset)
    init_batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (N,) + jnp.shape(x)), init_state)
    init_vec = RoboTaxiEnv.get_observation(init_state, env_params).as_vector()
    init_obs_batch = jnp.broadcast_to(init_vec, (N,) + init_vec.shape)
    key0 = jax.random.PRNGKey(0)  # the env is deterministic, the key is unused

    def env_step(state, action):
        obs, new_state, reward, done, info = RoboTaxiEnv.step(
            key0, state, action, env_params
        )
        return obs.as_vector(), new_state, reward, done, info["success"], info["collision"]

    v_step = jax.vmap(env_step)
    v_obs = jax.vmap(lambda s: RoboTaxiEnv.get_observation(s, env_params).as_vector())
    v_replan = jax.vmap(lambda s: RoboTaxiEnv.replan(s, env_params))

    # pool of random initial states used for autoreset (see make_spawn_pool)
    use_pool = cfg.p_default_start < 1.0
    if use_pool:
        pool = make_spawn_pool(env_params, init_state, cfg.spawn_pool, cfg.seed)
        pool_obs = jax.jit(v_obs)(pool)

    def init_carry(seed):
        return Carry(
            states=init_batch,
            obs=init_obs_batch,
            ep_ret=jnp.zeros(N),
            ep_len=jnp.zeros(N),
            rng=jax.random.PRNGKey(seed),
            t=jnp.asarray(0, dtype=jnp.int32),
        )

    # ---- one environment step for all N envs --------------------------------
    def body(params, stats, c: Carry, _):
        rng, k, k_idx, k_mix = jax.random.split(c.rng, 4)
        mean, log_std, value = net.apply(params, normalize(stats, c.obs))
        action = mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape)
        logp = gaussian_logprob(mean, log_std, action)

        next_obs, next_states, reward, done, success, crash = v_step(c.states, action)

        # refresh the DP paths of all envs once every K steps (one scalar
        # `cond`, so the DP is really skipped on the other steps)
        t1 = c.t + 1
        if K > 0:
            def do_replan(args):
                s, _o = args
                s = v_replan(s)
                return s, v_obs(s)

            next_states, next_obs = jax.lax.cond(
                t1 % K == 0, do_replan, lambda args: args, (next_states, next_obs)
            )

        # time-limit truncation is not a real terminal state: bootstrap from
        # the value of the last observation (the episode is cut by `done`)
        truncated = done & ~success & ~crash
        v_final = net.apply(params, normalize(stats, next_obs))[2]
        reward_train = reward * cfg.reward_scale + gamma * v_final * truncated

        ep_ret = c.ep_ret + reward
        ep_len = c.ep_len + 1.0

        # autoreset
        def pick(reset_leaf, new_leaf):
            d = done.reshape((-1,) + (1,) * (new_leaf.ndim - 1))
            return jnp.where(d, reset_leaf, new_leaf)

        if use_pool:
            idx = jax.random.randint(k_idx, (N,), 0, cfg.spawn_pool)
            use_default = jax.random.uniform(k_mix, (N,)) < cfg.p_default_start

            def pool_pick(pool_leaf, default_leaf):
                flag = use_default.reshape((-1,) + (1,) * (default_leaf.ndim - 1))
                return jnp.where(flag, default_leaf, pool_leaf[idx])

            reset_states = jax.tree.map(pool_pick, pool, init_batch)
            reset_obs = jnp.where(use_default[:, None], init_obs_batch, pool_obs[idx])
        else:
            reset_states, reset_obs = init_batch, init_obs_batch

        states_out = jax.tree.map(pick, reset_states, next_states)
        obs_out = jnp.where(done[:, None], reset_obs, next_obs)

        trans = Transition(c.obs, action, logp, value, reward_train, done)
        mets = dict(
            done=done.astype(jnp.float32),
            ret=jnp.where(done, ep_ret, 0.0),
            length=jnp.where(done, ep_len, 0.0),
            success=(done & success).astype(jnp.float32),
            crash=(done & crash).astype(jnp.float32),
        )
        c_out = Carry(
            states_out, obs_out,
            jnp.where(done, 0.0, ep_ret), jnp.where(done, 0.0, ep_len),
            rng, t1,
        )
        return c_out, (trans, mets)

    # ---- GAE ----------------------------------------------------------------
    def gae(trans: Transition, last_value):
        def f(carry, x):
            adv_next, v_next = carry
            reward, value, done = x
            nonterminal = 1.0 - done
            delta = reward + gamma * v_next * nonterminal - value
            adv = delta + gamma * lam * nonterminal * adv_next
            return (adv, value), adv

        _, advs = jax.lax.scan(
            f,
            (jnp.zeros_like(last_value), last_value),
            (trans.reward, trans.value, trans.done.astype(jnp.float32)),
            reverse=True,
        )
        return advs, advs + trans.value

    # ---- PPO loss -----------------------------------------------------------
    def loss_fn(params, mb: Batch, stats):
        mean, log_std, value = net.apply(params, normalize(stats, mb.obs))
        logp = gaussian_logprob(mean, log_std, mb.action)
        log_ratio = logp - mb.logp
        ratio = jnp.exp(log_ratio)
        adv = (mb.adv - mb.adv.mean()) / (mb.adv.std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv))
        v_clip = mb.value + jnp.clip(value - mb.value, -cfg.vf_clip, cfg.vf_clip)
        vl = 0.5 * jnp.mean(jnp.maximum((value - mb.target) ** 2, (v_clip - mb.target) ** 2))
        ent = gaussian_entropy(log_std)
        total = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        return total, jnp.stack([pg, vl, ent, approx_kl])

    # ---- one full PPO update ------------------------------------------------
    @jax.jit
    def update(ts: TrainState, stats: RunningStats, carry: Carry):
        params = ts.params
        carry, (trans, mets) = jax.lax.scan(
            lambda c, x: body(params, stats, c, x), carry, None, length=T
        )
        last_value = net.apply(params, normalize(stats, carry.obs))[2]
        adv, target = gae(trans, last_value)

        batch = Batch(
            obs=trans.obs.reshape(B, -1),
            action=trans.action.reshape(B, -1),
            logp=trans.logp.reshape(B),
            value=trans.value.reshape(B),
            adv=adv.reshape(B),
            target=target.reshape(B),
        )

        rng, k = jax.random.split(carry.rng)

        def epoch(ec, _):
            ts_, k_ = ec
            k_, kp = jax.random.split(k_)
            perm = jax.random.permutation(kp, B)
            shuffled = jax.tree.map(
                lambda x: x[perm].reshape((cfg.num_minibatches, -1) + x.shape[1:]), batch
            )

            def mb_step(ts__, mb):
                grads, aux = jax.grad(loss_fn, has_aux=True)(ts__.params, mb, stats)
                return ts__.apply_gradients(grads=grads), aux

            ts_, aux = jax.lax.scan(mb_step, ts_, shuffled)
            return (ts_, k_), aux

        (ts, _), aux = jax.lax.scan(epoch, (ts, k), None, length=cfg.epochs)
        aux = aux.mean(axis=(0, 1))  # (4,)

        new_stats = stats_update(stats, trans.obs.reshape(B, -1))
        carry = carry._replace(rng=rng)

        n_done = mets["done"].sum()
        denom = jnp.maximum(n_done, 1.0)
        metrics = dict(
            episodes=n_done,
            ret=mets["ret"].sum() / denom,
            length=mets["length"].sum() / denom,
            success=mets["success"].sum() / denom,
            crash=mets["crash"].sum() / denom,
            pg_loss=aux[0], v_loss=aux[1], entropy=aux[2], kl=aux[3],
            std=jnp.exp(params["params"]["log_std"]).mean(),
        )
        return ts, new_stats, carry, metrics

    return init_carry, update


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(path, params, stats: RunningStats):
    blob = dict(
        params=params,
        obs_mean=np.asarray(stats.mean),
        obs_var=np.asarray(stats.var),
        obs_count=np.asarray(stats.count),
    )
    with open(path, "wb") as f:
        f.write(to_bytes(blob))


def config_path_for(ckpt_path):
    base = os.path.basename(ckpt_path)
    for suffix in ("_best.msgpack", "_last.msgpack"):
        if base.endswith(suffix):
            return os.path.join(os.path.dirname(ckpt_path), base[: -len(suffix)] + ".json")
    return os.path.splitext(ckpt_path)[0] + ".json"


def load_checkpoint(ckpt_path):
    """Returns (net, params, stats, cfg, env_params, init_state)."""
    with open(config_path_for(ckpt_path)) as f:
        cfg = json.load(f)
    env_params, init_state = make_env(
        cfg["map_id"], cfg["max_steps"], cfg["action_repeat"], cfg["replan_period"],
        cfg["path_length"], cfg["num_rays"], cfg["perception_radius"],
        cfg.get("progress_on_path", False), cfg.get("plan_static_only", False),
        cfg.get("plug_slits", False),
    )
    obs_dim = RoboTaxiEnv.observation_dim(env_params)
    net = ActorCritic(hidden=cfg["hidden"])
    template = dict(
        params=net.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim))),
        obs_mean=np.zeros(obs_dim, np.float32),
        obs_var=np.ones(obs_dim, np.float32),
        obs_count=np.zeros((), np.float32),
    )
    with open(ckpt_path, "rb") as f:
        blob = from_bytes(template, f.read())
    stats = RunningStats(
        jnp.asarray(blob["obs_mean"]), jnp.asarray(blob["obs_var"]),
        jnp.asarray(blob["obs_count"]),
    )
    return net, blob["params"], stats, cfg, env_params, init_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # environment
    p.add_argument("--map", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400,
                   help="episode limit in ENV steps (400 x 4 / 60 Hz ~ 27 s)")
    p.add_argument("--action-repeat", type=int, default=4,
                   help="physics substeps per env step (4 -> 15 Hz control)")
    p.add_argument("--replan-period", type=int, default=8,
                   help="DP path refresh period in env steps (0 = legacy: DP inside step)")
    p.add_argument("--path-length", type=int, default=64)
    p.add_argument("--num-rays", type=int, default=32)
    p.add_argument("--perception-radius", type=float, default=5.0)
    p.add_argument("--plug-slits", action="store_true",
                   help="close zero-clearance single-cell gaps between two walls (map 2: the "
                        "slit over the wall in column 10), so the planner cannot route through them")
    p.add_argument("--plan-static-only", action="store_true",
                   help="the DP path ignores moving obstacles (needed on map 2, where the "
                        "path flips between two corridors otherwise)")
    p.add_argument("--progress-mode", choices=["path", "euclid"], default="path",
                   help="progress reward: along the DP path (default) or straight-line "
                        "distance to the goal (as in Assignment 1)")
    # PPO
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-clip", type=float, default=10.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=128)
    # misc
    p.add_argument("--spawn-pool", type=int, default=4096,
                   help="number of random initial states used for autoreset")
    p.add_argument("--p-default-start", type=float, default=0.25,
                   help="probability to restart from the real start (1.0 = never randomize)")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="continue training from this checkpoint (weights and observation "
                        "statistics are loaded, optimizer and lr schedule start anew; "
                        "use a lower --lr, e.g. 1e-4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--out-dir", type=str, default=os.path.join(ROOT_DIR, "checkpoints"))
    return p.parse_args()


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    prefix = os.path.join(cfg.out_dir, f"ppo_map{cfg.map}")

    env_params, init_state = make_env(
        cfg.map, cfg.max_steps, cfg.action_repeat, cfg.replan_period,
        cfg.path_length, cfg.num_rays, cfg.perception_radius,
        progress_on_path=(cfg.progress_mode == "path"),
        plan_static_only=cfg.plan_static_only, plug_slits=cfg.plug_slits,
    )
    obs_dim = RoboTaxiEnv.observation_dim(env_params)
    net = ActorCritic(hidden=cfg.hidden)

    steps_per_update = cfg.num_envs * cfg.rollout_len
    num_updates = max(cfg.total_timesteps // steps_per_update, 1)
    total_grad_steps = num_updates * cfg.epochs * cfg.num_minibatches
    schedule = optax.linear_schedule(cfg.lr, 0.0, total_grad_steps)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=schedule, eps=1e-5),
    )

    params = net.init(jax.random.PRNGKey(cfg.seed), jnp.zeros((1, obs_dim)))
    stats = stats_init(obs_dim)
    if cfg.init_ckpt:
        _, params, stats, init_cfg, _, _ = load_checkpoint(cfg.init_ckpt)
        assert init_cfg["hidden"] == cfg.hidden, "--hidden differs from the checkpoint"
        assert init_cfg["map_id"] == cfg.map, "the checkpoint was trained on another map"
        print(f"continuing from {cfg.init_ckpt}")
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    init_carry, update = build(env_params, init_state, net, cfg)
    carry = init_carry(cfg.seed)
    eval_fn = make_true_start_eval(net, env_params, init_state)

    with open(prefix + ".json", "w") as f:
        json.dump(dict(
            map_id=cfg.map, max_steps=cfg.max_steps, action_repeat=cfg.action_repeat,
            replan_period=cfg.replan_period, path_length=cfg.path_length,
            num_rays=cfg.num_rays, perception_radius=cfg.perception_radius,
            progress_on_path=(cfg.progress_mode == "path"),
            plan_static_only=cfg.plan_static_only,
            plug_slits=cfg.plug_slits, hidden=cfg.hidden, train_args=vars(cfg),
        ), f, indent=2)

    print(f"devices: {jax.devices()}")
    print(f"map {cfg.map} | obs dim {obs_dim} | {cfg.num_envs} envs x {cfg.rollout_len} steps "
          f"= {steps_per_update} steps/update | {num_updates} updates "
          f"({num_updates * steps_per_update:,} env steps)")

    log_file = open(prefix + "_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["update", "env_steps", "sps", "episodes", "return", "length",
                     "success", "crash", "pg_loss", "v_loss", "entropy", "kl", "std",
                     "eval_success", "eval_crash", "eval_d_goal", "eval_return", "eval_steps"])

    best_key = (-1, -1e9)
    t_start = time.time()
    t_last, steps_last = t_start, 0
    for u in range(1, num_updates + 1):
        ts, stats, carry, m = update(ts, stats, carry)

        if u == 1 or u % cfg.log_every == 0 or u == num_updates:
            m = jax.tree.map(float, jax.device_get(m))
            now = time.time()
            env_steps = u * steps_per_update
            if u == 1:
                # the first call includes compilation, report it separately
                print(f"compiled + first update: {now - t_start:.1f} s")
                t_last, steps_last = now, env_steps
                sps = float("nan")
            else:
                sps = (env_steps - steps_last) / max(now - t_last, 1e-9)
                t_last, steps_last = now, env_steps
            ev = eval_fn(ts.params, stats)  # deterministic episode from the real start
            print(f"upd {u:5d}/{num_updates} | steps {env_steps:>11,} | {sps:9.0f} sps | "
                  f"ret {m['ret']:8.1f} | len {m['length']:6.1f} | succ {m['success']:.2f} | "
                  f"crash {m['crash']:.2f} | ent {m['entropy']:.2f} | std {m['std']:.2f} | "
                  f"kl {m['kl']:.4f} || start: {'GOAL' if ev['success'] else ('crash' if ev['crash'] else 'timeout')}"
                  f" d_goal {ev['d_goal']:.1f} ret {ev['ret']:.1f}")
            writer.writerow([u, env_steps, sps, m["episodes"], m["ret"], m["length"],
                             m["success"], m["crash"], m["pg_loss"], m["v_loss"],
                             m["entropy"], m["kl"], m["std"],
                             int(ev["success"]), int(ev["crash"]), ev["d_goal"],
                             ev["ret"], ev["steps"]])
            log_file.flush()

            # "best" = reaches the goal from the real start (faster is better);
            # otherwise the closest to the goal without crashing
            score = (1, -ev["steps"]) if ev["success"] else (0, -ev["d_goal"] - 100.0 * ev["crash"])
            if score > best_key:
                best_key = score
                save_checkpoint(prefix + "_best.msgpack", ts.params, stats)
            t_last = time.time()  # do not count the evaluation in the sps

        if u % cfg.save_every == 0 or u == num_updates:
            save_checkpoint(prefix + "_last.msgpack", ts.params, stats)

    log_file.close()
    print(f"done in {time.time() - t_start:.0f} s | checkpoints: {prefix}_best/_last.msgpack")


if __name__ == "__main__":
    main()