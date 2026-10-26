"""Run the Astra + SAM3.1 auto-labeller end to end over trial100 intervals.

  detect   .venv-sam31  SAM3.1 'person' detections on every packet frame (GPU)
  astra    .venv        gpt-6-astra set-of-marks via logged-in Codex, N calls in parallel, usage-guarded
  fuse     .venv / .venv-sam31  instances (CPU) then resolve (GPU box/point masks) -> labels.json
  render   .venv        review contact sheets
Each stage skips intervals that already have its output, so the driver can be re-run after interruption.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from autolabel_common import EVIDENCE, ROOT, interval_dir

SCRIPTS = ROOT/'src'
PY = str(ROOT/'.venv/bin/python')
PY_SAM = str(ROOT/'.venv-sam31/bin/python')


def all_targets(trials=None):
    out = []
    for trial in sorted(EVIDENCE.glob('trial[0-9]*')) + sorted((ROOT/'evidence/autolabel-v2').glob('a2s[0-9]*')):
        if trials and trial.name not in trials:
            continue
        for pkt in sorted((trial/'selected-v1').glob('interval*/packet.json')):
            out.append((trial.name, pkt.parent.name))
    return out


def run(cmd):
    env = os.environ | {'PYTORCH_ALLOC_CONF': 'expandable_segments:True'}
    proc = subprocess.run(cmd, cwd=SCRIPTS, capture_output=True, text=True, env=env)
    lines = [l for l in (proc.stdout + proc.stderr).splitlines() if l.startswith('trial') or 'Error' in l or 'STOP' in l]
    print('\n'.join(lines), flush=True)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--parallel', type=int, default=3)
    ap.add_argument('--stages', default='detect,astra,fuse,render')
    ap.add_argument('--trials', nargs='*')
    ap.add_argument('targets', nargs='*', help='trialNNN/intervalNN (default: all trial100 intervals)')
    a = ap.parse_args()
    targets = [tuple(t.split('/')) for t in a.targets] or all_targets(a.trials)
    names = lambda ts: [f'{t}/{p}' for t, p in ts]
    stages = a.stages.split(',')
    print(f'{len(targets)} intervals, stages={stages}', flush=True)
    if 'detect' in stages:
        todo = [t for t in targets if not (interval_dir(a.run, *t)/'detections.json').exists()]
        if todo:
            run([PY_SAM, '-W', 'ignore', 'autolabel_detect.py', '--run', a.run] + names(todo))
    if 'astra' in stages:
        todo = [t for t in targets if not (interval_dir(a.run, *t)/'astra.json').exists()]
        failures = []
        with ThreadPoolExecutor(a.parallel) as pool:
            jobs = {pool.submit(run, [PY, 'autolabel_astra.py', '--run', a.run, n]): n for n in names(todo)}
            for job in as_completed(jobs):
                try:
                    job.result()
                except RuntimeError as error:
                    failures.append(jobs[job])
                    print(f'FAILED {jobs[job]}: {str(error)[:300]}', flush=True)
                    if 'STOP' in str(error):
                        for other in jobs:
                            other.cancel()
        if failures:
            print(f'astra failures ({len(failures)}): {failures}', flush=True)
    done = [t for t in targets if (interval_dir(a.run, *t)/'astra.json').exists()]
    if 'fuse' in stages and done:
        run([PY, 'autolabel_fuse.py', 'instances', '--run', a.run] + names(done))
        run([PY_SAM, '-W', 'ignore', 'autolabel_fuse.py', 'resolve', '--run', a.run] + names(done))
    if 'render' in stages:
        for t in done:
            if (interval_dir(a.run, *t)/'labels.json').exists() and not (interval_dir(a.run, *t)/'review').exists():
                run([PY, 'autolabel_render.py', '--run', a.run, f'{t[0]}/{t[1]}'])


if __name__ == '__main__':
    sys.exit(main())
