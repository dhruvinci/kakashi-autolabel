"""Keep every interval moving through the autolabel pipeline with one GPU job and a few Astra jobs at a time.

Per-interval state is read from files, so the orchestrator can be stopped and restarted at any time:
  detections.json -> astra.json -> frame_instances.json -> labels.json -> fight.json -> smooth.json -> verify.json
                                                                       -> dense/dense.json -> dense/pose2d.json
                                                                       -> body3d.json
GPU jobs (one at a time, model loaded once per batch), priority: detect > resolve > densify > pose2d > body3d.
Astra jobs (usage-guarded inside each script): identity > fight > verify, up to NET_JOBS concurrent processes.
Running jobs are recognised by ANCHORED command lines (never matches this process or a waiting shell).
Stop: touch autolabel/v1/orchestrator.stop
"""
import json
import subprocess
import time
from pathlib import Path

from autolabel_common import OUT, ROOT

SCRIPTS = ROOT/'src'
RUN = OUT/'v1'
LOG = RUN/'orchestrator.log'
STOP = RUN/'orchestrator.stop'
NET_JOBS = 6
GPU_BATCH = {'detect': 40, 'resolve': 30, 'densify': 6, 'pose2d': 20, 'pose2d_labels': 40, 'body3d': 10}
ENV = {'PYTORCH_ALLOC_CONF': 'expandable_segments:True'}


def targets():
    """Priority order (Astra budget is nearly spent): trial100, gi (new domain), continuous matches, then breadth."""
    out = []
    for d in sorted((ROOT/'evidence/trial100-v1').glob('trial[0-9]*/selected-v1/interval*')):
        out.append((d.parts[-3], d.name))
    for d in sorted((ROOT/'evidence/autolabel-v4').glob('a4g*/selected-v1/interval01')):
        out.append((d.parts[-3], d.name))
    for d in sorted((ROOT/'evidence/autolabel-v3').glob('a3m*/selected-v1/chunk*')):
        out.append((d.parts[-3], d.name))
    for d in sorted((ROOT/'evidence/autolabel-v2').glob('a2s*/selected-v1/interval01')):
        out.append((d.parts[-3], d.name))
    # second windows per source (after the 2026-10-08 usage reset), lowest priority: gi first, then breadth
    for root, pat in (('autolabel-v4', 'a4g*'), ('autolabel-v2', 'a2s*')):
        for d in sorted((ROOT/'evidence'/root).glob(f'{pat}/selected-v1/interval02')):
            out.append((d.parts[-3], d.name))
    return out


def has(t, rel):
    return (RUN/t[0]/t[1]/rel).exists()


def needs(t):
    """Next GPU stage and next Astra stage for one interval (None when nothing is due)."""
    gpu = net = None
    if not has(t, 'detections.json'):
        return 'detect', None
    if not has(t, 'astra.json'):
        net = 'astra'
    elif not has(t, 'labels.json'):
        gpu = 'resolve'
    else:
        if not has(t, 'fight.json'):
            net = 'fight'
        elif not has(t, 'verify.json') and has(t, 'smooth.json'):
            net = 'verify'
        if t[1] == 'interval02':
            # MVP (2026-10-09): second windows stay at 2 fps; skeletons on the label frames, then 3D bodies
            if not has(t, 'pose2d_labels.json'):
                gpu = 'pose2d_labels'
            elif not has(t, 'body3d.json'):
                gpu = 'body3d'
        elif not has(t, 'dense/dense.json'):
            gpu = 'densify'
        elif not has(t, 'dense/pose2d.json'):
            gpu = 'pose2d'
        elif not has(t, 'body3d.json'):
            gpu = 'body3d'
    return gpu, net


def running(pattern):
    r = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
    return len(r.stdout.split())


GPU_PATTERN = r'^(\.\./)?\.venv-(sam31|pose|s3d)/bin/python'
NET_PATTERN = r'^\.\./\.venv/bin/python (-u )?autolabel_(astra|fight|verify)\.py'


def running_net_targets():
    """(trial, packet) of Astra jobs already running, including ones started outside the orchestrator."""
    r = subprocess.run(['pgrep', '-af', NET_PATTERN], capture_output=True, text=True)
    out = set()
    for line in r.stdout.splitlines():
        last = line.split()[-1]
        if '/' in last:
            out.add(tuple(last.split('/', 1)))
    return out


MAX_AGE_S = {'net': 90*60, 'gpu': 4*3600}


def reap_stuck():
    """Kill pipeline jobs far past their legitimate runtime (e.g. hung after a power event or suspend).
    Net: 2 x 30 min codex timeouts + backoff < 90 min. GPU: largest batches take < 1 h."""
    import os
    import signal
    for kind, pattern in (('net', NET_PATTERN), ('gpu', GPU_PATTERN)):
        r = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True)
        for pid in r.stdout.split():
            try:
                age = int(subprocess.run(['ps', '-o', 'etimes=', '-p', pid], capture_output=True, text=True).stdout.strip() or 0)
            except ValueError:
                continue
            if age > MAX_AGE_S[kind]:
                children = subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True).stdout.split()
                for q in children + [pid]:
                    try:
                        os.kill(int(q), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                log(f'reaped stuck {kind} job pid {pid} after {age//60} min')


def log(msg):
    with LOG.open('a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def launch_gpu(stage, batch):
    names = [f'{a}/{b}' for a, b in batch]
    if stage == 'detect':
        cmd = ['../.venv-sam31/bin/python', '-u', '-W', 'ignore', 'autolabel_detect.py'] + names
    elif stage == 'resolve':
        subprocess.run(['../.venv/bin/python', 'autolabel_fuse.py', 'instances'] + names, cwd=SCRIPTS, capture_output=True)
        cmd = ['../.venv-sam31/bin/python', '-u', '-W', 'ignore', 'autolabel_fuse.py', 'resolve'] + names
    elif stage == 'densify':
        cmd = ['../.venv-sam31/bin/python', '-u', '-W', 'ignore', 'autolabel_densify.py'] + names
    elif stage == 'pose2d':
        cmd = ['../.venv-pose/bin/python', '-u', '-W', 'ignore', 'autolabel_pose2d.py'] + names
    elif stage == 'pose2d_labels':
        cmd = ['../.venv-pose/bin/python', '-u', '-W', 'ignore', 'autolabel_pose2d.py', '--label-frames'] + names
    else:
        cmd = ['../.venv-s3d/bin/python', '-u', '-W', 'ignore', 'autolabel_body3d.py'] + names
    out = open(RUN/f'orch-{stage}.log', 'a')
    subprocess.Popen(cmd, cwd=SCRIPTS, stdout=out, stderr=subprocess.STDOUT, env={**__import__('os').environ, **ENV})
    log(f'GPU {stage} x{len(batch)}: {names[0]}..{names[-1]}')


def launch_net(stage, t):
    script = {'astra': 'autolabel_astra.py', 'fight': 'autolabel_fight.py', 'verify': 'autolabel_verify.py'}[stage]
    out = open(RUN/f'orch-{stage}.log', 'a')
    post = None
    if stage == 'fight':  # smooth right after fight labels land (CPU, fast)
        post = f'../.venv/bin/python autolabel_smooth.py {t[0]}/{t[1]}'
    cmd = f'../.venv/bin/python -u {script} {t[0]}/{t[1]}' + (f' && {post}' if post else '')
    subprocess.Popen(['bash', '-c', cmd], cwd=SCRIPTS, stdout=out, stderr=subprocess.STDOUT)


ATTEMPTS = RUN/'orchestrator-attempts.json'
MAX_ATTEMPTS = 3


def load_attempts():
    return json.loads(ATTEMPTS.read_text()) if ATTEMPTS.exists() else {}


def main():
    log('orchestrator start')
    attempts = load_attempts()
    gpu_order = ['detect', 'resolve', 'pose2d', 'pose2d_labels', 'densify', 'body3d']  # pose2d first: it deletes dense frames (disk)
    net_order = ['astra', 'fight', 'verify']
    in_flight = {}
    while not STOP.exists():
        reap_stuck()
        ts = targets()
        states = {t: needs(t) for t in ts}
        if net_order and any('STOP:' in f.read_text(errors='ignore') for f in RUN.glob('orch-*.log')):
            log('Astra usage STOP seen; no further Astra jobs until Dhruva decides (delete the STOP lines to resume)')
            net_order = []
        if not running(GPU_PATTERN):
            for stage in gpu_order:
                batch = [t for t, (g, _) in states.items() if g == stage
                         and attempts.get(f'{stage}:{t[0]}/{t[1]}', 0) < MAX_ATTEMPTS][:GPU_BATCH[stage]]
                if batch:
                    for t in batch:
                        k = f'{stage}:{t[0]}/{t[1]}'
                        attempts[k] = attempts.get(k, 0) + 1
                    ATTEMPTS.write_text(json.dumps(attempts))
                    launch_gpu(stage, batch)
                    break
        free = NET_JOBS - running(NET_PATTERN)
        now = time.time()
        in_flight = {t: s for t, s in in_flight.items() if now - s < 3600}
        busy = running_net_targets()
        for stage in net_order:
            for t, (_, n) in states.items():
                if free <= 0:
                    break
                key = f'{stage}:{t[0]}/{t[1]}'
                if n == stage and t not in in_flight and t not in busy and attempts.get(key, 0) < MAX_ATTEMPTS:
                    attempts[key] = attempts.get(key, 0) + 1
                    ATTEMPTS.write_text(json.dumps(attempts))
                    launch_net(stage, t)
                    in_flight[t] = now
                    free -= 1
        pending = {s: sum(1 for g, n in states.values() if s in (g, n)) for s in gpu_order + ['astra', 'fight', 'verify']}
        (RUN/'orchestrator-state.json').write_text(json.dumps({'time': time.strftime('%H:%M:%S'), 'targets': len(ts), 'pending': pending}))
        time.sleep(45)
    log('orchestrator stop')


if __name__ == '__main__':
    main()
