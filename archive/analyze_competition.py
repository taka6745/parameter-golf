#!/usr/bin/env python3
"""Analyze all Parameter Golf PRs and generate COMPETITION_SCOPE.md"""

import json
import re
import os
from collections import Counter, defaultdict
from datetime import datetime

# ============================================================
# Load PR data
# ============================================================
open_prs = json.load(open('/tmp/paramgolf_open_prs.json'))
closed_prs = json.load(open('/tmp/paramgolf_closed_prs.json'))
all_prs = open_prs + closed_prs

print(f"Loaded {len(open_prs)} open + {len(closed_prs)} closed = {len(all_prs)} total PRs")

# Load local records
records = []
for track in ['records/track_10min_16mb', 'records/track_non_record_16mb']:
    if os.path.exists(track):
        for d in os.listdir(track):
            sj = os.path.join(track, d, 'submission.json')
            if os.path.exists(sj):
                with open(sj) as f:
                    rec = json.load(f)
                    rec['_dir'] = d
                    rec['_track'] = track
                    records.append(rec)
print(f"Loaded {len(records)} local records")

# ============================================================
# Parse scores from PR titles
# ============================================================
SCORE_PATTERNS = [
    r'val_bpb\s*[=:]\s*(\d+\.\d{2,6})',
    r'(\d+\.\d{2,6})\s*(?:val_)?[Bb][Pp][Bb]',
    r'[Bb][Pp][Bb]\s*[=:]*\s*(\d+\.\d{2,6})',
    r'(\d+\.\d{4,6})',  # bare 4+ decimal number as fallback
]

def extract_score(title):
    # Skip delta scores
    if re.search(r'[+-]\d+\.\d+\s*bpb', title, re.I):
        delta_match = re.search(r'[+-](\d+\.\d+)\s*bpb', title, re.I)
        # Still try to find an absolute score
    for pat in SCORE_PATTERNS:
        m = re.search(pat, title, re.I)
        if m:
            score = float(m.group(1))
            if 0.3 < score < 3.0:  # plausible BPB range
                return score
    return None

scored_prs = []
unscored_prs = []
for pr in all_prs:
    title = pr.get('title', '')
    score = extract_score(title)
    pr['_score'] = score
    pr['_author'] = pr.get('author', {}).get('login', 'unknown') if isinstance(pr.get('author'), dict) else str(pr.get('author', 'unknown'))
    pr['_date'] = pr.get('createdAt', '')[:10]
    pr['_merged'] = bool(pr.get('mergedAt'))
    pr['_state'] = 'merged' if pr['_merged'] else pr.get('state', 'OPEN')
    pr['_labels'] = [l.get('name', '') if isinstance(l, dict) else str(l) for l in pr.get('labels', [])]
    if score is not None:
        scored_prs.append(pr)
    else:
        unscored_prs.append(pr)

print(f"Scored: {len(scored_prs)}, Unscored: {len(unscored_prs)}")

# ============================================================
# Categorize techniques
# ============================================================
TECHNIQUE_KEYWORDS = {
    'SLOT': [r'\bSLOT\b', r'scored.position', r'per.sample.*delta'],
    'TTT': [r'\bTTT\b', r'test.time.train', r'score.first.*train'],
    'GPTQ': [r'\bGPTQ\b', r'hessian.*quant'],
    'XSA': [r'\bXSA\b', r'exclusive.*self.*att'],
    'NGRAM': [r'ngram', r'n-gram', r'bigram.*hash', r'BigramHash', r'EngramLite', r'backoff'],
    'LEAKYRELU': [r'LeakyReLU', r'leaky.*relu', r'relu.*sq'],
    'DEPTH_RECURRENCE': [r'depth.*recur', r'layer.*reuse', r'shared.*layer', r'U-?Net'],
    'SLIDING_WINDOW': [r'sliding.*window', r'stride.*\d+'],
    'SCYLLA_TOKENMONSTER': [r'Scylla', r'TokenMonster', r'byte260'],
    'EMA_SWA': [r'\bEMA\b', r'\bSWA\b', r'moving.*average'],
    'MUON': [r'\bMuon\b', r'MuonEq', r'NorMuon', r'Polar.*Express'],
    'TERNARY_BINARY': [r'ternary', r'1[.-]bit', r'binary.*quant', r'\b1b\b'],
    'INT6': [r'int6', r'Int6', r'6[.-]bit'],
    'INT5': [r'int5', r'Int5', r'5[.-]bit'],
    'INT4': [r'int4', r'Int4', r'4[.-]bit'],
    'LZMA': [r'LZMA', r'lzma'],
    'BROTLI': [r'Brotli', r'brotli'],
    'QAT': [r'\bQAT\b', r'quant.*aware'],
    'NOVEL_ARCH': [r'JEPA', r'\bMamba\b', r'Hymba', r'\bDEQ\b', r'diffusion', r'\bSSM\b', r'GatedDelta', r'Universal.*Trans'],
    'SP4096': [r'SP4096', r'sp4096', r'4096.*vocab'],
    'BPE8192': [r'BPE.?8192', r'bpe.?8192', r'8192.*vocab'],
    'PARALLEL_RESIDUAL': [r'parallel.*resid', r'parallel.*stream'],
    'SMEARGATE': [r'SmearGate', r'smear.*gate'],
    'COMPLEMENTARY': [r'complementary', r'comp.*train'],
    'LEGAL_TTT': [r'legal.*TTT', r'score.first'],
    'VRL': [r'\bVRL\b', r'value.*resid'],
    'ROPE': [r'RoPE', r'rotary', r'YaRN', r'Partial.*RoPE'],
    'FACTORIZED_EMBED': [r'factorized.*embed', r'ALBERT.*embed'],
    'FP8': [r'\bFP8\b', r'fp8'],
}

def categorize_pr(pr):
    title = pr.get('title', '')
    text = title  # could add body later
    techniques = []
    for tech, patterns in TECHNIQUE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.I):
                techniques.append(tech)
                break
    pr['_techniques'] = techniques

    # Legality flag
    title_lower = title.lower()
    if 'SLOT' in techniques:
        pr['_legality'] = 'CONTESTED_SLOT'
    elif 'NGRAM' in techniques and pr.get('_score') and pr['_score'] < 0.8:
        pr['_legality'] = 'LIKELY_INVALID_NGRAM'  # n-gram normalization bug
    elif pr.get('_score') and pr['_score'] < 0.5:
        pr['_legality'] = 'LIKELY_INVALID'  # physically implausible without bugs
    elif 'non-record' in title_lower or 'negative' in title_lower:
        pr['_legality'] = 'NON-RECORD'
    else:
        pr['_legality'] = 'LEGAL'

    return techniques

for pr in all_prs:
    categorize_pr(pr)

# ============================================================
# Analysis
# ============================================================

# Score leaderboard
scored_sorted = sorted(scored_prs, key=lambda p: p['_score'])

# Technique frequency
tech_counter = Counter()
tech_best_score = {}
tech_first_pr = {}
for pr in all_prs:
    for t in pr['_techniques']:
        tech_counter[t] += 1
        if pr['_score'] is not None:
            if t not in tech_best_score or pr['_score'] < tech_best_score[t]:
                tech_best_score[t] = pr['_score']
            if t not in tech_first_pr or pr['number'] < tech_first_pr[t]:
                tech_first_pr[t] = pr['number']

# Author analysis
author_counter = Counter()
author_best = {}
for pr in all_prs:
    a = pr['_author']
    author_counter[a] += 1
    if pr['_score'] is not None:
        if a not in author_best or pr['_score'] < author_best[a]:
            author_best[a] = pr['_score']

# Score distribution
score_bins = {'<0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0, '1.0-1.1': 0, '1.1-1.15': 0, '1.15-1.2': 0, '1.2-1.3': 0, '1.3+': 0}
for pr in scored_prs:
    s = pr['_score']
    if s < 0.7: score_bins['<0.7'] += 1
    elif s < 0.9: score_bins['0.7-0.9'] += 1
    elif s < 1.0: score_bins['0.9-1.0'] += 1
    elif s < 1.1: score_bins['1.0-1.1'] += 1
    elif s < 1.15: score_bins['1.1-1.15'] += 1
    elif s < 1.2: score_bins['1.15-1.2'] += 1
    elif s < 1.3: score_bins['1.2-1.3'] += 1
    else: score_bins['1.3+'] += 1

# Timeline by week
weekly = defaultdict(list)
for pr in scored_prs:
    week = pr['_date'][:7]  # YYYY-MM
    weekly[week].append(pr['_score'])

# Merged records
merged_prs = [pr for pr in all_prs if pr['_merged']]

# Legal-only leaderboard (exclude SLOT, invalid n-gram, implausible scores)
legal_scored = [pr for pr in scored_prs if pr['_legality'] == 'LEGAL' and pr['_score'] and pr['_score'] > 0.8]
legal_sorted = sorted(legal_scored, key=lambda p: p['_score'])

# Contested
contested_scored = [pr for pr in scored_prs if pr['_legality'] == 'CONTESTED']

# ============================================================
# Generate COMPETITION_SCOPE.md
# ============================================================
out = []
out.append("# Parameter Golf Competition Scope Analysis")
out.append(f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
out.append(f"## Data: {len(open_prs)} open + {len(closed_prs)} closed = {len(all_prs)} total PRs, {len(records)} merged records\n")

# Executive Summary
out.append("## 1. Executive Summary\n")
out.append(f"- **Total PRs**: {len(all_prs)} ({len(open_prs)} open, {len(closed_prs)} closed, {len(merged_prs)} merged)")
out.append(f"- **PRs with parseable scores**: {len(scored_prs)} ({len(scored_prs)*100//len(all_prs)}%)")
out.append(f"- **Merged SOTA**: {min(r.get('val_bpb', 99) for r in records if 'val_bpb' in r):.4f} BPB" if records else "- No records found")
if legal_sorted:
    out.append(f"- **Best legal open PR**: {legal_sorted[0]['_score']:.4f} BPB (PR #{legal_sorted[0]['number']} by @{legal_sorted[0]['_author']})")
if scored_sorted:
    out.append(f"- **Best ANY open PR**: {scored_sorted[0]['_score']:.4f} BPB (PR #{scored_sorted[0]['number']} by @{scored_sorted[0]['_author']})")
out.append(f"- **Contested (SLOT) PRs**: {len(contested_scored)}")
out.append(f"- **Our position**: 1.7292 BPB (Mac, 1000 steps, int8 quantized)\n")

# Score Leaderboard
out.append("## 2. Score Leaderboard\n")

out.append("### 2a. Merged Records (ground truth)\n")
out.append("| # | BPB | Author | Technique Summary | Date |")
out.append("|---|-----|--------|-------------------|------|")
for i, r in enumerate(sorted(records, key=lambda x: x.get('val_bpb', 99))):
    bpb = r.get('val_bpb', 'N/A')
    author = r.get('author', 'unknown')
    blurb = r.get('blurb', r.get('name', ''))[:60]
    date = r.get('date', '')[:10]
    out.append(f"| {i+1} | {bpb} | {author} | {blurb} | {date} |")

out.append(f"\n### 2b. Top 50 Legal Open PRs\n")
out.append("| Rank | PR# | BPB | Author | Title (truncated) | Techniques |")
out.append("|------|-----|-----|--------|-------------------|------------|")
for i, pr in enumerate(legal_sorted[:50]):
    title = pr.get('title', '')[:50]
    techs = ', '.join(pr['_techniques'][:4])
    out.append(f"| {i+1} | #{pr['number']} | {pr['_score']:.4f} | @{pr['_author']} | {title} | {techs} |")

out.append(f"\n### 2c. Top 30 Contested (SLOT) PRs\n")
out.append("| Rank | PR# | BPB | Author | Title (truncated) |")
out.append("|------|-----|-----|--------|-------------------|")
for i, pr in enumerate(sorted(contested_scored, key=lambda p: p['_score'])[:30]):
    title = pr.get('title', '')[:55]
    out.append(f"| {i+1} | #{pr['number']} | {pr['_score']:.4f} | @{pr['_author']} | {title} |")

# Score Distribution
out.append("\n## 3. Score Distribution\n")
out.append("| BPB Range | Count | % |")
out.append("|-----------|-------|---|")
for rng, cnt in score_bins.items():
    pct = cnt * 100 // max(len(scored_prs), 1)
    bar = '#' * (pct // 2)
    out.append(f"| {rng} | {cnt} | {pct}% {bar} |")

# Technique Taxonomy
out.append("\n## 4. Technique Frequency\n")
out.append("| Technique | PRs Using | Best Score | First PR |")
out.append("|-----------|-----------|------------|----------|")
for tech, count in tech_counter.most_common():
    best = f"{tech_best_score.get(tech, 'N/A'):.4f}" if tech in tech_best_score else "N/A"
    first = f"#{tech_first_pr.get(tech, '?')}" if tech in tech_first_pr else "?"
    out.append(f"| {tech} | {count} | {best} | {first} |")

# Legality Analysis
out.append("\n## 5. Legality Analysis\n")
legal_count = sum(1 for pr in all_prs if pr['_legality'] == 'LEGAL')
contested_count = sum(1 for pr in all_prs if pr['_legality'] == 'CONTESTED')
nonrecord_count = sum(1 for pr in all_prs if pr['_legality'] == 'NON-RECORD')
out.append(f"- **Legal**: {legal_count} PRs ({legal_count*100//len(all_prs)}%)")
out.append(f"- **Contested (SLOT)**: {contested_count} PRs ({contested_count*100//len(all_prs)}%)")
out.append(f"- **Non-record/negative**: {nonrecord_count} PRs")
out.append(f"\n**The SLOT divide**: All scores below ~1.00 BPB use SLOT. Legal-only frontier is ~{legal_sorted[0]['_score']:.2f} BPB." if legal_sorted else "")

# Author Analysis
out.append("\n## 6. Top Authors\n")
out.append("### By PR count\n")
out.append("| Author | PRs | Best Score |")
out.append("|--------|-----|------------|")
for author, count in author_counter.most_common(30):
    best = f"{author_best.get(author, 'N/A'):.4f}" if author in author_best else "N/A"
    out.append(f"| @{author} | {count} | {best} |")

# Novel Approaches
out.append("\n## 7. Novel Approaches\n")
novel_prs = [pr for pr in all_prs if 'NOVEL_ARCH' in pr['_techniques']]
out.append(f"**{len(novel_prs)} PRs with non-transformer architectures:**\n")
out.append("| PR# | BPB | Author | Title |")
out.append("|-----|-----|--------|-------|")
for pr in sorted(novel_prs, key=lambda p: p.get('_score') or 99)[:20]:
    score = f"{pr['_score']:.4f}" if pr['_score'] else "N/A"
    title = pr.get('title', '')[:60]
    out.append(f"| #{pr['number']} | {score} | @{pr['_author']} | {title} |")

# Gap Analysis
out.append("\n## 8. Gap Analysis\n")
out.append("### Techniques from our research NOT seen in PRs:\n")
our_techniques = [
    'Signed hashing for n-gram tables',
    'Distributional Categories (DC500/DC1000)',
    'English Knowledge Engine (POS, capitalization, context)',
    'Skip-bigram table (prev2→next)',
    'Tabulation hashing (XOR lookup)',
    'Perfect hashing (bbhash)',
    'NorMuon + MuonEq-R combined',
    'Trimmed mean loss (trim tails)',
    'Rho-1 excess loss with n-gram reference',
    'Dual-codebook n-gram compression',
    'Dendritic MLP (block-diagonal)',
    'Codon-style eval tokenization search',
    'Online eval n-gram mixing (backward-looking)',
    'Predictive coding gate (suppress predictable)',
    'Folding shard ordering',
    'FIM-diverse shard selection',
]
for t in our_techniques:
    out.append(f"- {t}")

out.append("\n### Underexplored in competition (low PR count):\n")
low_count = [(t, c) for t, c in tech_counter.items() if c < 5]
for t, c in sorted(low_count, key=lambda x: x[1]):
    out.append(f"- {t}: only {c} PRs")

# Strategic Recommendations
out.append("\n## 9. Strategic Recommendations\n")
out.append("### Safest path to beat SOTA:")
out.append("- Stack proven winners: 11L/3xMLP + XSA + NorMuon + BigramHash + GPTQ int6 + sliding window")
out.append("- Add our unique edge: n-gram logit bias + DC500 + signed hashing + skip-bigram")
out.append("- Legal eval: temperature scaling + online n-gram cache + legal TTT\n")
out.append("### Highest-ceiling path:")
out.append("- Everything above + Meta-TTT (FOMAML) + parallel TTT search on 8 GPUs")
out.append("- Factorized embedding → extra layers")
out.append("- Rho-1 excess loss training\n")
out.append("### Most novel/differentiated path:")
out.append("- Our n-gram bias stack (nobody else has DC categories + skip-bigram + signed hashing)")
out.append("- Dendritic MLP + predictive coding gate")
out.append("- Perfect hashing for zero-collision n-grams")
out.append("- Codon-style eval tokenization search")

# Write output
content = '\n'.join(out)
with open('COMPETITION_SCOPE.md', 'w') as f:
    f.write(content)

print(f"\nWrote COMPETITION_SCOPE.md ({len(content)} bytes, {len(out)} lines)")
print(f"\nKey stats:")
print(f"  Total PRs: {len(all_prs)}")
print(f"  With scores: {len(scored_prs)}")
print(f"  Best legal: {legal_sorted[0]['_score']:.4f} (PR #{legal_sorted[0]['number']})" if legal_sorted else "  No legal scored PRs")
print(f"  Best any: {scored_sorted[0]['_score']:.4f} (PR #{scored_sorted[0]['number']})" if scored_sorted else "  No scored PRs")
print(f"  Most popular technique: {tech_counter.most_common(1)[0][0]} ({tech_counter.most_common(1)[0][1]} PRs)")
print(f"  Most prolific author: {author_counter.most_common(1)[0][0]} ({author_counter.most_common(1)[0][1]} PRs)")
