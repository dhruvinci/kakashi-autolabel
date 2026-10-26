"""Screen discovered candidates from the gi queries into an acquisition import (sport='gi').

Added 2026-10-04 when Dhruva approved gi BJJ (the original selection excludes gi on purpose). Only candidates
found by a gi query are considered; no-gi, MMA, other sports and non-action titles are rejected. Organisation-specific title terms come from the
selection config (see select_sources.py); the published code keeps only generic terms.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from select_sources import _any, load_config

ROOT = Path(__file__).resolve().parents[1]
GI_QUERIES = {q['query'] for q in json.loads((ROOT/'manifests/discovery-queries-gi.json').read_text())}


CFG = load_config()


def screen(item, cfg=CFG):
    t = (item.get('title') or '').casefold()
    if not (item.get('duration') and 90 <= item['duration'] <= 3600) or not item.get('channel_id'):
        return None, 'duration_or_channel'
    if re.search(r'\bno[\s-]?gi\b|\bnogi\b|\bmma\b|kickboxing|muay thai|boxing|\bjudo\b|wrestling match|sambo|'
                 + _any(cfg['nogi_title_terms'] + cfg['other_scope_terms'] + cfg.get('mma_title_terms', [])), t):
        return None, 'not_gi_scope'
    if re.search(r'podcast|interview|reaction|vlog|highlight|compilation|top \d|q&a|trailer|seminar recap|vs no.?gi|\bgi vs\b', t):
        return None, 'non_action_title'
    queries = [q for q in item['discovery_evidence'] if q['query'] in GI_QUERIES]
    if not queries:
        return None, 'not_from_gi_query'
    context = Counter(q['context'] for q in queries).most_common(1)[0][0]
    if re.search(r'tutorial|technique|how to|instruction|\bdrill|lesson|\bguide|concept|\bclass\b', t):
        context = 'tutorial'
    elif re.search(r'full match|\bvs\.?\b|final|championship|' + _any(cfg.get('gi_competition_terms', [])), t):
        context = 'competition'
    elif re.search(r'rolling|\broll|sparring|rounds|open mat', t):
        context = 'training'
    return {'url': f"https://www.youtube.com/watch?v={item['id']}", 'title': item['title'], 'sport': 'gi',
            'context': context, 'ready': True,
            'metadata': {'expected_channel_id': item['channel_id'], 'channel_name': item.get('channel'),
                         'estimated_duration_s': item['duration'], 'scope_review': 'gi metadata screened; visual pick pending',
                         'discovery_evidence': queries, 'split': 'unassigned_pending_source_grouping',
                         'rights_status': 'not_cleared_for_redistribution'}}, 'accepted'


def main(per_cell=80):
    groups, rejected = defaultdict(list), Counter()
    for line in (ROOT/'manifests/discovered-candidates.jsonl').read_text().splitlines():
        row, why = screen(json.loads(line))
        (groups[row['context']].append(row) if row else rejected.update([why]))
    selected, channel_count = [], Counter()
    for ctx in sorted(groups):
        rows = sorted(groups[ctx], key=lambda r: r['url'])
        for r in rows:
            ch = r['metadata']['expected_channel_id']
            if channel_count[ch] >= 12:  # channel cap keeps source diversity
                continue
            if sum(s['context'] == ctx for s in selected) >= per_cell:
                break
            channel_count[ch] += 1
            selected.append(r)
    for i, r in enumerate(selected):
        r['priority'] = 5000 - i  # above the queued mma/nogi backlog
    (ROOT/'manifests/gi-selected.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    print(json.dumps({'selected': len(selected), 'by_context': dict(Counter(r['context'] for r in selected)),
                      'rejected': dict(rejected)}, indent=1))


if __name__ == '__main__':
    main()
