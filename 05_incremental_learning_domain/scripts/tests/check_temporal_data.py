"""测试 1/3：领域增量数据完整性检查（D1~D6 × train/test）。

领域线的数据契约与时间线不同：
  - 每个阶段是一个领域（D1~D6），阶段内部再覆盖 T1~T5 全部年代；
  - era 字段记录年代（T1..T5），period 字段记录领域任务名；
  - A3：文档不要求互斥，段落内容严格互斥。
因此实际检查委托给 prepare/validate_domain_tasks.py（配额/字段/标签/互斥/年份/跨实验重叠）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PREPARE = Path(__file__).resolve().parents[1] / "prepare"
sys.path.insert(0, str(PREPARE))

import validate_domain_tasks  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("测试 1/3：D1-D6 领域增量数据完整性检查")
    print("=" * 78)
    return validate_domain_tasks.main()


if __name__ == "__main__":
    sys.exit(main())
