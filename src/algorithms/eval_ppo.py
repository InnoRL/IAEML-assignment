"""Evaluate a trained PPO policy on the robotaxi environment and record a video.

Run from the repository root:

    # a checkpoint from any folder
    python src/algorithms/eval_ppo.py --ckpt checkpoints/map0_random/ppo_map0_best.msgpack

    # or just the run folder (+ --map, --which best|last), and open the video when done
    python src/algorithms/eval_ppo.py --run-dir checkpoints/map0_random --which last --open

The video is written to ``videos/<run folder>_<checkpoint>.mp4`` unless ``--out`` is given,
so different runs do not overwrite each other.

The policy is deterministic (the mean of the Gaussian) and the environment is
deterministic as well, so one episode is enough. The same action repeat and
path-refresh period as in training are used (taken from the config json that
``train_ppo.py`` writes next to the checkpoint).
"""

import argparse
import os
import shutil
import sys
from types import SimpleNamespace

import jax
import numpy as np
from jax import numpy as jnp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# make `env`, `utils`, `demo_robotaxi`, `algorithms` (src/) importable from anywhere
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from algorithms.train_ppo import load_checkpoint, normalize  # noqa: E402
from demo_robotaxi import _draw_frame, _save_stills  # noqa: E402  (renderer of Assignment 1)
from env.robotaxi import RoboTaxiEnv  # noqa: E402

KEY = jax.random.PRNGKey(0)


def find_ffmpeg():
    """Path to ffmpeg: the system one, or the binary bundled with imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def run_episode(ckpt, deterministic=True, seed=0):
    net, params, stats, cfg, p, s0 = load_checkpoint(ckpt)
    K = int(cfg["replan_period"])

    @jax.jit
    def policy(vec, rng):
        mean, log_std, _ = net.apply(params, normalize(stats, vec))
        if deterministic:
            return jnp.clip(mean, -1.0, 1.0)
        return jnp.clip(mean + jnp.exp(log_std) * jax.random.normal(rng, mean.shape), -1, 1)

    obs, state = RoboTaxiEnv.reset(KEY, p, s0)
    rng = jax.random.PRNGKey(seed)
    frames, ret, done, t = [], 0.0, False, 0
    info = {}
    while not done:
        rng, k = jax.random.split(rng)
        action = policy(obs.as_vector(), k)
        obs, state, reward, done, info = RoboTaxiEnv.step(KEY, state, action, p)
        t += 1
        if K > 0 and t % K == 0:  # same path refresh schedule as in training
            state = RoboTaxiEnv.replan(state, p)
            obs = RoboTaxiEnv.get_observation(state, p)
        ret += float(reward)
        done = bool(done)
        frames.append((state, obs, float(reward)))

    stats_out = dict(
        map=cfg["map_id"], env_steps=len(frames),
        sim_seconds=len(frames) * cfg["action_repeat"] / p.fps,
        success=bool(info["success"]), crash=bool(info["collision"]),
        ret=ret, d_goal=float(info["distance_to_goal"]),
    )
    return p, cfg, frames, stats_out


def record_video(p, cfg, frames, path):
    """Real-time video: every env step lasts action_repeat / fps seconds."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("ffmpeg not found: no video written. Install it with `pip install imageio-ffmpeg` "
              "(or `winget install Gyan.FFmpeg` and restart the terminal).")
        return
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # _draw_frame shows t / p.fps seconds; time counts env steps here
    p_draw = SimpleNamespace(
        map_height_width=p.map_height_width, fps=p.fps / cfg["action_repeat"]
    )
    step_dt = cfg["action_repeat"] / p.fps
    video_fps = 30
    repeat = max(int(round(step_dt * video_fps)), 1)  # frames per env step

    fig = plt.figure(figsize=(12.0, 6.4), dpi=100)
    ax = fig.add_subplot(111)
    writer = FFMpegWriter(
        fps=video_fps,
        metadata={"title": "robotaxi PPO", "artist": "IAEML Assignment 2"},
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23"],
    )
    ret = 0.0
    with writer.saving(fig, path, dpi=100):
        for state, obs, reward in frames:
            ret += reward
            _draw_frame(ax, p_draw, state, obs, ret)
            for _ in range(repeat):
                writer.grab_frame()
    plt.close(fig)
    print(f"video written: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="path to a .msgpack checkpoint")
    ap.add_argument("--run-dir", default=None,
                    help="folder with the checkpoints of a run (instead of --ckpt)")
    ap.add_argument("--map", type=int, default=0, help="map of the run (with --run-dir)")
    ap.add_argument("--which", choices=["best", "last"], default="best",
                    help="which checkpoint of the run to use (with --run-dir)")
    ap.add_argument("--out", default=None,
                    help="output mp4 (default <repo>/videos/<run folder>_<checkpoint>.mp4)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--open", action="store_true", help="open the video when it is written")
    ap.add_argument("--stills-dir", default=None)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    args = ap.parse_args()

    if args.ckpt:
        ckpt = args.ckpt
    elif args.run_dir:
        ckpt = os.path.join(args.run_dir, f"ppo_map{args.map}_{args.which}.msgpack")
    else:
        ap.error("give --ckpt or --run-dir")
    if not os.path.isfile(ckpt):
        ap.error(f"checkpoint not found: {ckpt}")
    print(f"checkpoint: {ckpt}")

    p, cfg, frames, st = run_episode(ckpt, deterministic=not args.stochastic)
    print(st)
    if args.stills_dir:
        p_draw = SimpleNamespace(
            map_height_width=p.map_height_width, fps=p.fps / cfg["action_repeat"]
        )
        _save_stills(p_draw, frames, args.stills_dir)
    if not args.no_video:
        folder = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        out = args.out or os.path.join(ROOT_DIR, "videos", f"{folder}_{stem}.mp4")
        record_video(p, cfg, frames, out)
        if args.open and os.path.isfile(out):
            if hasattr(os, "startfile"):  # Windows
                os.startfile(out)
            else:
                print(f"open it with your video player: {out}")