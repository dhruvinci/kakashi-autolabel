"""Stage 7: temporal smoothing of per-frame categorical labels (CPU).

Viterbi decoding of a hidden Markov model per interval:
- position (Kakashi L1A) from fight.json. Transition prior: stay, allowed BJJ transitions, rare everything else.
  Emission: Astra's label weighted by its stated confidence, with extra mass on its second choice.
- posture per tracked actor from labels.json (taxonomy posture); sticky prior, uniform switches.
Writes smooth.json next to labels.json. The raw labels stay untouched; smoothed values are added as separate fields.
"""
import argparse
import json

import numpy as np

from autolabel_common import interval_dir, write_json

POSITIONS = ['standing', 'guard', 'side_control', 'mount', 'back', 'turtle', 'leg_entanglement', 'transition', 'not_applicable', 'unknown']
EDGES = {('standing', x) for x in ('guard', 'side_control', 'mount', 'back', 'turtle', 'leg_entanglement')} | {
    ('guard', 'side_control'), ('guard', 'mount'), ('guard', 'back'), ('guard', 'leg_entanglement'), ('guard', 'turtle'),
    ('side_control', 'mount'), ('side_control', 'back'), ('side_control', 'turtle'), ('mount', 'back'),
    ('back', 'turtle'), ('leg_entanglement', 'turtle')}
POSTURES = ['standing', 'crouched_or_kneeling', 'seated', 'supine', 'prone', 'side_lying', 'all_fours',
            'airborne_or_inverted', 'occluded', 'unknown']
CONF = {'high': 0.85, 'medium': 0.7, 'low': 0.5}


def position_transitions(stay=0.9, allowed=0.08, rare=0.004):
    n = len(POSITIONS)
    T = np.full((n, n), rare)
    for i, a in enumerate(POSITIONS):
        for j, b in enumerate(POSITIONS):
            if (a, b) in EDGES or (b, a) in EDGES or {'not_applicable', 'unknown', 'transition'} & {a, b}:
                T[i, j] = allowed
        T[i, i] = 0
        T[i] = T[i]/T[i].sum()*(1 - stay)
        T[i, i] = stay
    return T


def sticky_transitions(n, stay=0.85):
    T = np.full((n, n), (1 - stay)/(n - 1))
    np.fill_diagonal(T, stay)
    return T


def emission(states, label, confidence=None, alt=None):
    n = len(states)
    e = np.full(n, 1.0)
    if label in states:
        c = CONF.get(confidence, 0.7)
        e = np.full(n, (1 - c)/(n - 1))
        e[states.index(label)] = c
        if alt in states and alt != label:
            e[states.index(alt)] += (1 - c)/2
    return e/e.sum()


def viterbi(T, E):
    """T: [n, n] transitions; E: [t, n] emissions. Returns the most likely state index path."""
    t, n = E.shape
    logT, logE = np.log(T), np.log(E)
    score = np.log(np.full(n, 1/n)) + logE[0]
    back = np.zeros((t, n), int)
    for k in range(1, t):
        cand = score[:, None] + logT
        back[k] = cand.argmax(0)
        score = cand.max(0) + logE[k]
    path = [int(score.argmax())]
    for k in range(t - 1, 0, -1):
        path.append(int(back[k, path[-1]]))
    return path[::-1]


TOP_POSITIONS = {'guard', 'side_control', 'mount', 'back', 'turtle'}


def consistent_detail_top(frames, path, reach=3):
    """Detail (L1B) and top actor that agree with the smoothed position. Where smoothing changed the position, borrow
    them from the nearest frame whose raw position equals the smoothed one (within reach), else detail 'unknown'.
    Illegal pairs (a detail not listed for the position) become 'unknown'; top is None for non-top positions."""
    from autolabel_fight import L1B
    detail, top = [], []
    for k, pos in enumerate(path):
        src = frames[k] if frames[k]['position'] == pos else next(
            (frames[j] for d in range(1, reach + 1) for j in (k - d, k + d)
             if 0 <= j < len(frames) and frames[j]['position'] == pos), None)
        d = src['position_detail'] if src else 'unknown'
        if d not in L1B.get(pos, []) and d not in ('none', 'unknown'):
            d = 'unknown'
        if pos in L1B and d == 'none':
            d = 'unknown'
        if pos not in L1B:
            d = 'none'
        t = (src or frames[k]).get('top_actor') if pos in TOP_POSITIONS else None
        if pos == 'leg_entanglement':
            t = (src or frames[k]).get('top_actor')
        detail.append(d)
        top.append(t)
    return detail, top


def changes(seq):
    return sum(a != b for a, b in zip(seq, seq[1:]))


def segmented(T, E, cuts):
    """Viterbi run independently on each shot (no smoothing across camera cuts)."""
    path, bounds = [], [0] + sorted(c for c in cuts if 0 < c < len(E)) + [len(E)]
    for a, b in zip(bounds, bounds[1:]):
        if b > a:
            path += viterbi(T, E[a:b])
    return path


def smooth_interval(base):
    from autolabel_densify import shot_cuts
    out = {'version': 2, 'stage': 'hmm_viterbi_smoothing'}
    labels = json.loads((base/'labels.json').read_text())
    cuts = shot_cuts([f['path'] for f in labels['frames']])
    out['shot_cuts'] = cuts  # label-frame indices that start a new shot
    if (base/'fight.json').exists():
        fight = json.loads((base/'fight.json').read_text())
        frames = fight['result']['frames']
        raw = [f['position'] for f in frames]
        E = np.stack([emission(POSITIONS, f['position'], f['confidence'], f['position_alt']) for f in frames])
        path = [POSITIONS[i] for i in segmented(position_transitions(), E, cuts)]
        detail, top = consistent_detail_top(frames, path)
        out['position'] = {'frames': [f['frame'] for f in frames], 'raw': raw, 'smoothed': path, 'detail': detail, 'top': top,
                           'raw_changes': changes(raw), 'smoothed_changes': changes(path),
                           'frames_changed': sum(a != b for a, b in zip(raw, path))}
    actors = [a['actor_id'] for a in labels['actors'] if a['role'] in ('athlete', 'instructor', 'referee')]
    out['posture'] = {}
    for actor in actors:
        raw = [next((i['posture'] for i in f['instances'] if i['actor_id'] == actor), None) for f in labels['frames']]
        E = np.stack([emission(POSTURES, p) if p else np.full(len(POSTURES), 1/len(POSTURES)) for p in raw])
        path = [POSTURES[i] for i in segmented(sticky_transitions(len(POSTURES)), E, cuts)]
        out['posture'][actor] = {'raw': raw, 'smoothed': path, 'raw_changes': changes([p for p in raw if p]),
                                 'smoothed_changes': changes(path),
                                 'frames_changed': sum(1 for a, b in zip(raw, path) if a and a != b)}
    write_json(base/'smooth.json', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    for t in a.targets:
        trial_id, packet_id = t.split('/')
        o = smooth_interval(interval_dir(a.run, trial_id, packet_id))
        pos = o.get('position', {})
        print(t, 'position changes', pos.get('raw_changes'), '->', pos.get('smoothed_changes'), 'frames changed', pos.get('frames_changed'),
              '| posture changes', {k: (v['raw_changes'], v['smoothed_changes']) for k, v in o['posture'].items()})
