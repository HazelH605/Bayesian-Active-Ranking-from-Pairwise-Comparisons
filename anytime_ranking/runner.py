"""Run independent episodes and resume completed or interrupted work."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import csv
import gzip
import json
import os
from . import __version__
from .checkpointing import atomic_json
from .config import _validate, scientific_config, scenario_policy_indices
from .experiment import run_episode
from .reporting import make_reports

def _episode_name(scenario_index, policy_index, replicate):
    return f"s{scenario_index:03d}_p{policy_index:03d}_r{replicate:05d}.json.gz"

def _run_task(config, destination, key):
    si, pi, rep = key
    name = _episode_name(si, pi, rep)
    query = destination / "query_checkpoints" / name.removesuffix(".gz")
    result = run_episode(config, config.scenarios[si], config.experiment.policies[pi], si, pi, rep,
                         query_checkpoint_path=query)
    completed = destination / "episode_checkpoints" / name
    temporary = completed.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        json.dump(dict(schema=2, key=key, run=result.run, trajectory=result.trajectory), handle)
    os.replace(temporary, completed)
    # Remove the active checkpoint after the completed episode has been saved.
    query.unlink(missing_ok=True)
    return key

def _assemble(destination, keys, config, elapsed):
    # Assemble outputs one episode at a time.
    diagnostic_columns = ["baseline_phase", "baseline_round", "quicksort_pass", "acquisition_score",
                          "spr_selected_probability", "spr_max_probability", "spr_entropy"]
    with (destination/"runs.csv.tmp").open("w", newline="", encoding="utf-8") as rh, \
         gzip.open(destination/"trajectories.csv.gz.tmp", "wt", newline="", encoding="utf-8", compresslevel=3) as th:
        rw = tw = None
        for key in keys:
            path = destination / "episode_checkpoints" / _episode_name(*key)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("schema") != 2 or tuple(value["key"]) != key:
                raise ValueError(f"incompatible episode checkpoint: {path}")
            if rw is None:
                rw = csv.DictWriter(rh, fieldnames=list(value["run"]))
                columns = list(value["trajectory"][-1]) + diagnostic_columns
                tw = csv.DictWriter(th, fieldnames=columns, extrasaction="ignore")
                rw.writeheader()
                tw.writeheader()
            rw.writerow(value["run"])
            tw.writerows(value["trajectory"])
    os.replace(destination/"runs.csv.tmp", destination/"runs.csv")
    os.replace(destination/"trajectories.csv.gz.tmp", destination/"trajectories.csv.gz")
    atomic_json(destination/"manifest.json", dict(package_version=__version__, schema=2,
        created_utc=datetime.now(timezone.utc).isoformat(), wall_seconds_last_session=elapsed,
        n_episodes=len(keys), config=config.to_dict()))
    make_reports(destination)

def run_resumable(config, destination):
    """Hold an operating-system lock while updating a result directory."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is using {destination}") from error
        return _run_resumable(config, destination)


def _run_resumable(config, destination):
    _validate(config)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "resume_state.json"
    identity = json.loads(json.dumps(scientific_config(config)))
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if old.get("schema") != 2 or old["scientific_config"] != identity:
            raise ValueError("configuration changed: choose a new output directory")
    elif any(path.name != ".run.lock" for path in destination.iterdir()):
        raise ValueError("output directory is not an experiment from this version; choose a new directory")
    for name in ("episode_checkpoints", "query_checkpoints"):
        (destination/name).mkdir(exist_ok=True)
    keys = [(si, pi, rep) for si, s in enumerate(config.scenarios)
            for pi in scenario_policy_indices(config, s) for rep in range(config.experiment.replicates)]
    pending = [key for key in keys if not (destination/"episode_checkpoints"/_episode_name(*key)).exists()]
    completed = len(keys)-len(pending)
    failures = []

    def progress(status):
        atomic_json(state_path, dict(schema=2, scientific_config=identity, status=status,
                    completed=completed, total=len(keys), failures=failures))

    def finished(key, error=None):
        nonlocal completed
        if error is None:
            completed += 1
            print(f"Completed {completed}/{len(keys)} episodes", flush=True)
        else:
            failures.append(dict(key=key, error=f"{type(error).__name__}: {error}"))
            print(f"Failed {key}: {error}; other episodes continue", flush=True)
        progress("running")

    progress("running")
    print(f"Recovered {completed}/{len(keys)} completed episodes", flush=True)
    started = perf_counter()
    if config.experiment.jobs == 1:
        for key in pending:
            try:
                _run_task(config, destination, key)
            except Exception as error:
                finished(key, error)
            else:
                finished(key)
    elif pending:
        with ProcessPoolExecutor(max_workers=config.experiment.jobs) as pool:
            futures = {pool.submit(_run_task, config, destination, key): key for key in pending}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                except Exception as error:
                    finished(key, error)
                else:
                    finished(key)
    if failures:
        progress("failed")
        raise RuntimeError("some episodes failed; inspect resume_state.json and rerun the same command")
    _assemble(destination, keys, config, perf_counter()-started)
    progress("complete")
    return destination
