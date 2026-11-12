# -*- coding: utf-8 -*-
"""classify_domains.py 的离线自测（不调 API）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify_domains import parse_answer, build_user_prompt, SYSTEM_PROMPT

tests = [
    ('{"domain": "经济与金融", "confidence": 0.9}', ('经济与金融', 0.9)),
    ('好的，分类结果是：{"domain": "自然与生态环境", "confidence": 0.85}',
     ('自然与生态环境', 0.85)),
    ('{"domain": "交通运输建设", "confidence": 1.2}', ('无法归类', 1.0)),
    ('{"domain": "随便", "confidence": "x"}', ('无法归类', 0.0)),
    ('{"domain":"法律与政治监管","confidence":0.75}', ('法律与政治监管', 0.75)),
]
fails = 0
for text, want in tests:
    got = parse_answer(text)
    ok = got == want
    fails += 0 if ok else 1
    print('OK ' if ok else 'FAIL', got)
print('parse tests:', 'ALL PASS' if fails == 0 else f'{fails} FAILED')

print('system prompt len:', len(SYSTEM_PROMPT))
rec = json.loads(
    (Path(__file__).parent.parent.parent / "data" / "domain"
     / "domain_classification_input.jsonl").read_text(encoding="utf-8").splitlines()[0])
print('--- 样例候选 ---')
print('Title:', rec['Title'])
print('Dept:', rec['IssuingDepartment'], '| era:', rec['era'],
      '| nR:', rec['n_R_paras'], '| hit:', rec['hit_domains'])
print('--- user prompt ---')
print(build_user_prompt(rec))
