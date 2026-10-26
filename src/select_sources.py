"""Conservative metadata screening; its outputs still require later visual review.

Channel allowlists and organisation-specific title terms are read from a JSON config (default
manifests/selection-config.json; see src/config/selection-config.example.json). The lists used for the original
catalogue named specific official channels and events and are not published; without a config only the generic
title rules below apply.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path(os.environ.get('KAKASHI_SELECTION_CONFIG',ROOT/'manifests'/'selection-config.json'))


def load_config(path=CONFIG):
    """mma_channels / nogi_channels: {channel_id: name}; training_channels / grappling_channels: [channel_id];
    nogi_title_terms / other_scope_terms / excluded_title_patterns: lists of regular expressions."""
    cfg={'mma_channels':{},'nogi_channels':{},'training_channels':[],'grappling_channels':[],
         'nogi_title_terms':[],'other_scope_terms':[],'excluded_title_patterns':[]}
    if Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


CFG=load_config()


def _any(terms):
    return '|'.join(terms) if terms else r'(?!x)x'   # an empty list matches nothing


def screen(item,cfg=None):
    cfg=CFG if cfg is None else cfg
    MMA_CHANNELS,NOGI_CHANNELS=cfg['mma_channels'],cfg['nogi_channels']
    title=item.get('title') or ''
    t=title.casefold().replace('–','-').replace('—','-')
    duration=item.get('duration')
    channel=item.get('channel_id')
    if not duration or not 90<=duration<=3600:
        return None,'duration_outside_90s_to_60min_or_unknown'
    if not channel:
        return None,'channel_identity_unresolved'
    without_nogi=re.sub(r'\bno[\s-]?gi\b|\bnogi\b','',t)
    if re.search(r'\bgi\b|kimono',without_nogi):
        return None,'gi_or_mixed_gi_title'
    if re.search(r'kickboxing|muay thai|\bboxing\b|\bjudo\b|street fight|self.defen[cs]e|'+_any(cfg['other_scope_terms']),t):
        return None,'other_sport_or_mixed_scope_title'
    if re.search(r'podcast|interview|press conference|weigh.in|trailer|highlight|compilation|best knockouts|top \d|reaction|vlog|q&a|\bgi vs\b|\d+ submissions from|\badvice\b|shares his approach|\bdummy\b|wing chun|\bgameplay\b',t):
        return None,'mostly_non_action_or_montage_title'
    if re.search(_any(cfg['excluded_title_patterns']),t):
        return None,'known_mixed_rules_bout'
    nogi=bool(re.search(r'\bno[\s-]?gi\b|\bnogi\b|submission grappling|submission wrestling|'+_any(cfg['nogi_title_terms']),t))
    mma=bool(re.search(r'\bmma\b|mixed martial',t))
    if nogi:
        sport='nogi'
    elif mma:
        sport='mma'
    elif channel in NOGI_CHANNELS:
        sport='nogi'
    elif channel in MMA_CHANNELS and re.search(r'\bvs?\.?\b|full fight|free fight',t):
        sport='mma'
    elif channel in cfg['grappling_channels'] and re.search(r'grappling|\bbjj\b',t):
        sport='nogi'
    else:
        return None,'sport_not_established_by_title_or_channel'
    if re.search(r'full fight|full match|free fight|championship|\bfinals?\b',t) and not re.search(r'breakdown|analysis|tutorial',t):
        context='competition'
    elif re.search(r'tutorial|technique|how to|instruction|\bescape|\bdefen[cs]e|\bdrill|lesson|\bguide|\btips|concept|\bclass\b',t):
        context='tutorial'
    elif re.search(r'rolling|\broll\b|\brolls\b|sparring|training|gym rounds|full round',t):
        context='training'
    elif channel in cfg['training_channels']:
        context='training'
    elif re.search(r'full fight|full match|free fight|\bvs?\.?\b|championship|\bfinals?\b',t):
        context='competition'
    elif re.search(r'guard|passing|sweep|armbar|heel hook|takedown|back take|leg lock|triangle|choke|wrestling',t):
        context='tutorial'
    else:
        return None,'context_or_action_relevance_unresolved'
    reasons=['explicit_sport_in_title' if (nogi or mma) else 'specialist_channel_or_grappling_title',f'context_title:{context}']
    priority=60
    if 180<=duration<=1200:
        priority+=10
    if context in ('training','tutorial'):
        priority+=10
    if re.search(r'escape|defen[cs]e|counter|mistake|transition|women|beginner',t):
        priority+=5
    return {'url':f"https://www.youtube.com/watch?v={item['id']}",'title':title,
            'sport':sport,'context':context,'priority':priority,'ready':True,
            'metadata':{'expected_channel_id':channel,'channel_name':item.get('channel'),
                        'estimated_duration_s':duration,'scope_review':'metadata_screened; visual_review_pending',
                        'selection_reasons':reasons,'discovery_evidence':item['discovery_evidence'],
                        'split':'unassigned_pending_source_grouping','rights_status':'not_cleared_for_redistribution'}},'accepted_provisional'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=1000)
    args=parser.parse_args()
    groups=defaultdict(list)
    rejected=Counter()
    for line in (ROOT/'manifests'/'discovered-candidates.jsonl').read_text().splitlines():
        row,reason=screen(json.loads(line))
        if row:
            groups[(row['sport'],row['context'])].append(row)
        else:
            rejected[reason]+=1
    for rows in groups.values():
        rows.sort(key=lambda r:(-r['priority'],r['url']))
    # Round robin gives scarce sport/context cells early access to the cache.
    selected=[]
    while len(selected)<args.limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected)<args.limit:
                selected.append(groups[key].pop(0))
    for i,row in enumerate(selected):
        row['priority']=1000-i
    dest=ROOT/'manifests'/'selected-sources.jsonl'
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    summary={'selected_provisional':len(selected),'sport_context':dict(Counter(f"{r['sport']}/{r['context']}" for r in selected)),
             'rejected_or_held':dict(rejected),'remaining_accepted':sum(map(len,groups.values()))}
    (ROOT/'receipts'/'selection-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
