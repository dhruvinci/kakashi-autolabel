"""Draw every figure in docs/assets/ from numbers recorded in the research workspace (matplotlib only).

No video frames, crops or masks are drawn: the source footage is third-party, so the figures are diagrams and charts.
Each number below carries the workspace file and line it comes from ("ws:" = the private research workspace, which
is not published; "repo:" = this repository).

  python tools/make_figures.py            # writes docs/assets/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO/'docs/assets'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993'
BG, INK, INK2, GRID = '#fcfcfb', '#2b2b29', '#5c5b57', '#e6e5e1'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'font.size': 10,
    'axes.edgecolor': GREY, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'regular', 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'legend.frameon': False, 'legend.labelcolor': INK2,
})


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS/name, dpi=160, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(name, (ASSETS/name).stat().st_size // 1024, 'KB')


def hbars(ax, labels, values, colors, fmt, xmax=None, xlabel=''):
    """Horizontal bars from zero, value at the tip, first label on top."""
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, height=0.55, color=colors, zorder=2)
    ax.set_yticks(y, labels)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, xmax or max(values)*1.18)
    if ax.get_xlim()[1] >= 1000:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    for yy, v in zip(y, values):
        ax.text(v + (ax.get_xlim()[1]*0.012), yy, fmt(v), va='center', color=INK, fontsize=9)
    ax.set_xlabel(xlabel)


# ----------------------------------------------------------------------------------------------- diagrams

def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.12', linewidth=0,
                                facecolor=color, alpha=0.13, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), 0.07, h, boxstyle='square,pad=0', linewidth=0, facecolor=color, zorder=2))
    ax.text(x + 0.2, y + h - 0.22, title, ha='left', va='top', fontsize=10.5, color=INK, weight='bold', zorder=3)
    ax.text(x + 0.2, y + h - 0.62, sub, ha='left', va='top', fontsize=8.6, color=INK2, zorder=3, linespacing=1.35)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=GREY, linewidth=1.2, zorder=0,
                                 shrinkA=2, shrinkB=2))


def pipeline():
    """README figure: the stages, coloured by where they run (repo: src/autolabel_orchestrate.py docstring)."""
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 12.8)
    ax.axis('off')
    W, H = 4.9, 2.25
    rows = [
        (9.8, [('1  Acquire', 'public videos -> catalogue\n<=720p masters; subclips\nare metadata only', GREY),
               ('2  Pick window', 'Astra picks a 20 s\nhigh-action interval;\n2 fps evidence packet', ORANGE),
               ('3  Detect', 'SAM 3.1 "person" on\nevery frame (~41)', BLUE),
               ('4  Set-of-marks', 'Astra sees numbered marks:\nwho is who, role, posture,\nmerged / missed people', ORANGE),
               ('5  Resolve', 'Astra boxes + points ->\nSAM 3.1 masks; one owner\nper pixel; review flags', BLUE)]),
        (6.2, [('6  Fight', 'Astra: position, detail,\ntop / dominant actor,\ntimed events', ORANGE),
               ('7  Smooth', 'HMM / Viterbi over\nposition and posture;\nno smoothing across cuts', GREY),
               ('8  Verify', 'Astra second opinion on\nflagged frames -> tier;\ndisputes -> uncertain', ORANGE),
               ('9  Densify', 'SAM 3.1 video tracker:\n2 -> 12.5 fps masks;\nheld-out anchors', BLUE),
               ('10  Pose + body', 'ViTPose+ 2D (mask-guided);\nSAM 3D Body 3D joints', BLUE)]),
        (2.6, [('11  Strikes (MMA)', 'wrist / ankle speed peaks\n-> Astra judges 6-frame\nstrips', ORANGE),
               ('12  Mask fix', 'dense mask replaces\nspeckled fallback masks\nwhen coherent', BLUE),
               ('13  Export', 'COCO instances + frame,\ninterval, event, strike\nJSONL; quality tiers', GREY),
               ('Reviews', 'AI audits (Claude, Astra);\nDhruva\'s gold review;\nVLM saturation bench', AQUA)]),
    ]
    xs = [0.2 + i*5.35 for i in range(5)]
    for y, stages in rows:
        for i, (t, s, c) in enumerate(stages):
            box(ax, xs[i], y, W, H, t, s, c)
            if i:
                arrow(ax, (xs[i-1] + W, y + H/2), (xs[i], y + H/2))
    # row wraps
    arrow(ax, (xs[4] + W/2, 9.8), (xs[0] + W/2, 6.2 + H))
    arrow(ax, (xs[4] + W/2, 6.2), (xs[0] + W/2, 2.6 + H))
    ax.text(0.2, 12.55, 'Auto-labeller: GPT-6 Astra (set-of-marks) + local SAM 3.1, one 20 s interval at a time',
            fontsize=13, color=INK, va='top')
    for x, c, t in [(0.25, BLUE, 'local model on a 6 GB GPU'), (6.6, ORANGE, 'Astra call (Codex CLI, usage-guarded)'),
                    (15.2, GREY, 'CPU'), (18.0, AQUA, 'review')]:
        ax.add_patch(FancyBboxPatch((x, 0.75), 0.45, 0.45, boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor=c, linewidth=0))
        ax.text(x + 0.65, 0.97, t, va='center', fontsize=9.5, color=INK2)
    save(fig, 'pipeline.png')


def workflow_split():
    """E02: who does what. Roboflow's documented Auto Label vs this pipeline (ws: docs/autolabel/README.md:3-8)."""
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8.6)
    ax.axis('off')
    W, H = 6.6, 2.1
    ax.text(0.2, 8.4, 'Two ways to split the work between a VLM and SAM', fontsize=13, color=INK, va='top')
    ax.text(0.2, 7.25, "Roboflow Auto Label (documented workflow)", fontsize=10.5, color=INK2, va='center')
    box(ax, 0.2, 4.6, W, H, 'VLM proposes', 'class boxes from the\nimage (Astra)', GREY)
    box(ax, 8.6, 4.6, W, H, 'SAM 3 refines', 'one mask per\nproposed box', GREY)
    box(ax, 17.0, 4.6, W, H, 'Output', 'boxes / polygons for the\nprompted classes', GREY)
    arrow(ax, (0.2 + W, 5.65), (8.6, 5.65))
    arrow(ax, (8.6 + W, 5.65), (17.0, 5.65))
    ax.text(0.2, 3.6, 'This pipeline (reversed)', fontsize=10.5, color=INK2, va='center')
    box(ax, 0.2, 0.95, W, H, 'SAM 3.1 finds everyone', 'every person, every frame,\nlocally (no hosted inference)', BLUE)
    box(ax, 8.6, 0.95, W, H, 'Astra reads the marks', 'who is who across frames,\nrole, posture, merges, misses,\nownership points', ORANGE)
    box(ax, 17.0, 0.95, W, H, 'Output', 'instance masks with identity,\nrole, posture, position,\nevents, review flags', GREY)
    arrow(ax, (0.2 + W, 2.0), (8.6, 2.0))
    arrow(ax, (8.6 + W, 2.0), (17.0, 2.0))
    save(fig, 'workflow_split.png')


# ----------------------------------------------------------------------------------------------- charts

def catalogue():
    """E00 funnel. ws: manifests/discovered-candidates.jsonl (3,518 lines); manifests/catalog.sqlite `sources`
    (1,233 rows, 908 status=complete), queried 2026-10-26; repo: docs/export_summary.json:9 (670 sources)."""
    fig, ax = plt.subplots(figsize=(8, 2.9))
    hbars(ax, ['unique candidates\n(metadata search)', 'catalogue entries\n(screened)', 'verified masters\n(downloaded)',
               'sources with\nauto-labels'], [3518, 1233, 908, 670], [GREY, GREY, BLUE, BLUE],
          lambda v: f'{v:,}', xmax=4100, xlabel='videos')
    ax.set_title('From search results to labelled sources')
    save(fig, 'chart_catalogue.png')


def effort():
    """E01. ws: docs/PILOT_REPORT.md:49-53 (8 excerpts) and :55-56 (low-effort repeat, 4 excerpts)."""
    fig, ax = plt.subplots(figsize=(8, 3.0))
    labels = ['low', 'medium', 'high', 'low, repeat']
    supported = [90/90, 102/104, 125/127, 47/47]
    checklist = [50/54, 52/54, 52/54, 19/27]
    y = list(range(len(labels)))[::-1]
    for yy, s, c in zip(y, supported, checklist):
        ax.plot([min(s, c), max(s, c)], [yy, yy], color=GRID, linewidth=3, zorder=1)
    ax.scatter(supported, y, s=90, color=GREY, zorder=3, edgecolors=BG, linewidths=2, label='claims supported')
    ax.scatter(checklist, y, s=90, color=BLUE, zorder=3, edgecolors=BG, linewidths=2, label='checklist items covered')
    ax.text(19/27 - 0.012, y[3], '19/27', ha='right', va='center', fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.6, 1.03)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis='x')
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('share (AI reviewer, not human ground truth; axis starts at 60%)')
    ax.set_title('Agent annotation effort: claims are right, coverage is what varies')
    ax.legend(loc='upper left', ncol=1, fontsize=9)
    save(fig, 'chart_effort.png')


def calibration():
    """E03. ws: docs/autolabel/CALIBRATION.md:28 and :31 (same 508 frames)."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    hbars(ax, ['agent annotation\n(trial, sparse boxes)', 'auto-labeller\n(prompt v3, masks)'], [43, 508], [GREY, BLUE],
          lambda v: f'{v} / 508', xmax=620, xlabel='frames with person geometry')
    ax.set_title('Same 508 calibration frames: frames with any person geometry')
    save(fig, 'chart_calibration.png')


def flags_vs_audit():
    """E04. ws: reviews/autolabel-v1/REVIEW.md:39-40."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    hbars(ax, ['rated good', 'rated usable with filtering', 'rated poor'], [0.14, 0.36, 0.67], [BLUE, BLUE, BLUE],
          lambda v: f'{v:.0%}', xmax=0.8, xlabel='share of frames carrying a review or risk flag')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title('Flags track the AI audit\'s interval rating (42 audited intervals)')
    save(fig, 'chart_flags_vs_audit.png')


def adjudication():
    """E05. ws: reviews/autolabel-v1/REVIEW.md:99-102 (30 mask and 30 position disputes)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 2.7), gridspec_kw={'wspace': 0.75})
    hbars(a1, ['Astra right', 'pipeline flag right'], [20, 10], [BLUE, GREY], lambda v: f'{v}', xmax=32,
          xlabel='disputes (of 30)')
    a1.set_title('Masks: Astra "fine", pipeline flagged')
    hbars(a2, ['Astra right', 'pipeline right', 'both wrong', 'unsure'], [8, 6, 6, 10], [BLUE, GREY, GREY, GREY],
          lambda v: f'{v}', xmax=32, xlabel='disputes (of 30)')
    a2.set_title('Positions: Astra vs pipeline label')
    fig.suptitle('Blind adjudication by three Claude reviewers (AI judgement, not ground truth)', x=0.02, ha='left',
                 y=1.08, fontsize=12, color=INK)
    save(fig, 'chart_adjudication.png')


def frame_rate():
    """E06. ws: reviews/autolabel-v1/REVIEW.md:139-142."""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    hbars(ax, ['median frame', 'fastest 10% of motion\n(median)', 'worst 10% of frames', 'worst 10%, in\ntransitions / scrambles'],
          [3.4, 8.9, 10.4, 13.2], [GREY, BLUE, BLUE, BLUE], lambda v: f'{v}%', xmax=15.5,
          xlabel='2D keypoint error, % of body height')
    ax.set_title('Interpolating 2 fps keypoints vs measuring at 12.5 fps')
    save(fig, 'chart_frame_rate.png')


def cohort_masks():
    """E07. ws: reviews/autolabel-v1/REVIEW.md:188-194 (833 actor-frames, Claude reviewers)."""
    rows = [('all (833)', 85.1, 9.6, 5.3), ('breadth MMA / no-gi (278)', 84, 13, 4), ('gi (346)', 84, 8, 7),
            ('full matches (125)', 89, 8, 3), ('first 100 sources (84)', 87, 7, 6)]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    y = list(range(len(rows)))[::-1]
    for yy, (lab, g, p, w) in zip(y, rows):
        tot = g + p + w   # rounded shares; normalise so the bar ends at 100%
        left = 0
        for v, c in ((g, BLUE), (p, ORANGE), (w, GREY)):
            ax.barh(yy, v/tot*100 - 0.4, left=left, height=0.55, color=c, zorder=2)
            left += v/tot*100
        ax.text(g/tot*100/2, yy, f'{g:g}%', ha='center', va='center', color='white', fontsize=9, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of audited actor-frames')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY)],
              labels=['good', 'partial', 'wrong or missing'], loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3,
              fontsize=9)
    ax.set_title('Mask quality by cohort, full-corpus AI audit')
    save(fig, 'chart_cohort_masks.png')


def strikes():
    """E08. repo: docs/export_summary.json:68-83 (= ws: reviews/autolabel-v1/REVIEW.md:265-281)."""
    s = json.loads((REPO/'docs/export_summary.json').read_text())['strikes_by_type']
    labels = list(s)[:15]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    hbars(ax, [k.replace('_', ' ') for k in labels], [s[k] for k in labels], [BLUE]*len(labels), lambda v: f'{v}',
          xmax=270, xlabel='strikes accepted by Astra (1,322 of 3,175 candidates)')
    ax.set_title('Strike pass: accepted strikes by type (AI labels)')
    save(fig, 'chart_strikes.png')


def gold():
    """E09. ws: reviews/autolabel-v1/REVIEW.md:315-321 (Dhruva's first gold session, `unclear` excluded)."""
    rows = [('identity (misses were all mask faults)', 12, 19), ('events', 12, 18), ('strikes', 13, 20),
            ('position class', 14, 20), ('position + detail + top', 11, 20), ('mask pairs (both masks good)', 12, 23)]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    hbars(ax, [r[0] for r in rows], [r[1]/r[2] for r in rows], [BLUE]*len(rows), lambda v: f'{v:.0%}', xmax=1.0,
          xlabel='fully correct, as judged by Dhruva')
    for yy, (lab, k, n) in zip(list(range(len(rows)))[::-1], rows):
        ax.text(0.985, yy, f'{k}/{n}', ha='right', va='center', fontsize=9, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title('First human check: Dhruva\'s gold review (103 verdicts, 47 notes)')
    save(fig, 'chart_gold.png')


def vlm_bench():
    """E10. ws: reviews/autolabel-v1/REVIEW.md:341-345 (positions, 17 cases) and :372-374 (events, 16 cases)."""
    pos = [('Astra high, frame + verify', 14, 'A'), ('Astra xhigh, 4 s strip', 14, 'A'), ('Astra high, frame', 13, 'A'),
           ('Astra xhigh, frame', 13, 'A'), ('Astra high, strip (outlines)', 13, 'A'), ('Astra high, strip (tags)', 12, 'A'),
           ('Claude Opus, frame', 9, 'C'), ('Claude Opus, + verify', 8, 'C'), ('Claude Opus, strip', 6, 'C'),
           ('Claude Opus, tags', 5, 'C')]
    ev = [('Astra high, whole 20 s at 2 fps', 13, 'A'), ('Astra high, event +/-4 s at 4 fps', 9, 'A'),
          ('Claude Opus, event +/-4 s at 4 fps', 7, 'C')]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.2), gridspec_kw={'height_ratios': [10, 3.4], 'hspace': 0.55})
    col = {'A': BLUE, 'C': ORANGE}
    hbars(a1, [p[0] for p in pos], [p[1] for p in pos], [col[p[2]] for p in pos], lambda v: f'{v}/17', xmax=17,
          xlabel='positions fully right (of 17; 8 hard)')
    a1.set_title('VLM saturation bench: positions')
    hbars(a2, [e[0] for e in ev], [e[1] for e in ev], [col[e[2]] for e in ev], lambda v: f'{v}/16', xmax=16,
          xlabel='events fully right: type + outcome + actor (of 16; 4 hard)')
    a2.set_title('Events')
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)], labels=['GPT-6 Astra', 'Claude Opus'],
              loc='lower right', fontsize=9)
    save(fig, 'chart_vlm_bench.png')


def maskfix_ab():
    """E11. ws: reviews/autolabel-v1/maskfix/README.md:44 (Dhruva's blind A/B, 32 dense-replaced frames)."""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    hbars(ax, ['fixed mask better', 'original better', 'same', 'both bad'], [9, 4, 13, 6], [BLUE, GREY, GREY, GREY],
          lambda v: f'{v}', xmax=16, xlabel='frames (of 32), judged blind by Dhruva')
    ax.set_title('Mask fix: dense video masks vs original fallback masks')
    save(fig, 'chart_maskfix_ab.png')


def tiers():
    """README / docs/EXPORT.md. repo: docs/export_summary.json:29-35."""
    t = json.loads((REPO/'docs/export_summary.json').read_text())['frames_by_quality_tier']
    order = [('clean', 'clean'), ('astra_verified', 'Astra-verified'), ('dense_replaced_unverified', 'dense mask, unverified'),
             ('rejected', 'rejected'), ('flagged_unverified', 'flagged, unverified')]
    fig, ax = plt.subplots(figsize=(8, 2.7))
    hbars(ax, [o[1] for o in order], [t[o[0]] for o in order], [BLUE, BLUE, ORANGE, GREY, GREY], lambda v: f'{v:,}',
          xmax=57000, xlabel='frames in the export (65,159)')
    ax.set_title('Export autolabel-v1: frames by quality tier')
    save(fig, 'chart_tiers.png')


if __name__ == '__main__':
    pipeline()
    workflow_split()
    catalogue()
    effort()
    calibration()
    flags_vs_audit()
    adjudication()
    frame_rate()
    cohort_masks()
    strikes()
    gold()
    vlm_bench()
    maskfix_ab()
    tiers()
