"""
5_model/experiments/phase1_lab/build_specific_component_groups.py

v5 전용 — v3의 연결요소(Union-Find, survivor+attached)를 v4의 "특이(specific)" HI만
남기고 필터링해서, GroupedHardConcreteGate용 group_ids(scen_gate_width=len(specific_idx)
기준 로컬 인덱스)를 만든다.

왜 성장 그룹(seg_s_groups)을 그대로 못 쓰는가(docs/260901_V5_DESIGN.md §2-2/§3-2 참고):
성장 그룹은 편상관계수로 "일부러 서로 안 겹치는" 조합을 고른 것이라 GroupedHardConcreteGate가
전제하는 "그래디언트 비슷함(=원시 상관 높음)"과 반대다. 연결요소(성장 그룹 + attached를
합친 것)만이 진짜 |raw corr|>=threshold인 덩어리라 이 전제에 맞는다.

왜 공유(shared) HI는 뺴는가: 공유 25개는 이미 shared_gate 하나로 더 강하게 묶여있어서
여기 또 넣을 필요가 없다(중복 규제). 연결요소에 공유 HI가 섞여 있으면 그 부분만 버리고
특이 HI만 남긴다 — 필터링 후 멤버가 1개면 사실상 "그룹 없음"과 동일하게 취급된다.

출력 group_ids의 순서는 SCRModel의 _specific_idx(=torch.nonzero(~shared_hi_mask), 오름차순
원본 인덱스 순)와 반드시 일치해야 한다 — 이 스크립트도 concepts_in_order를 그대로 오름차순
순회해서 만들므로 자동으로 맞는다(순서가 어긋나면 조용히 엉뚱한 그룹으로 묶여 학습되는
"티 안 나는 버그"가 되므로, main()에서 자체 정합성 검증을 반드시 거친다).

사용 예:
  python 5_model/experiments/phase1_lab/build_specific_component_groups.py \
      --v3-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_v3.json \
      --interaction-json 5_model/experiments/phase1_lab/results/hi_scenario_interaction_k25_full_N2.json \
      --tag k25_full_N2_v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402

RESULTS_DIR = _HERE / "results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v5용 — 연결요소를 specific HI로 필터링한 group_ids 생성")
    p.add_argument("--v3-groups-json", required=True, dest="v3_groups_json",
                   help="build_synergy_groups.py --global-dedup 산출물(seg_s_groups+seg_s_group_attached 필요)")
    p.add_argument("--interaction-json", required=True, dest="interaction_json",
                   help="test_hi_scenario_interaction.py 산출물(공유/특이 분류)")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _build_components(groups: list[list[int]], attached: list[list[int]], n_hi: int) -> dict[int, int]:
    """comp_of[raw_hi_idx] = component_id. groups[gi]+attached[gi]가 component gi를 이룬다.
    n_hi개 전부가 정확히 한 번씩만 나와야 한다(안 그러면 v3 산출물 자체가 깨진 것 —
    조용히 넘기지 않고 바로 에러)."""
    comp_of: dict[int, int] = {}
    for gi, (g, a) in enumerate(zip(groups, attached)):
        for idx in g + a:
            if idx in comp_of:
                raise ValueError(f"HI 인덱스 {idx}가 두 연결요소({comp_of[idx]}, {gi})에 동시에 속합니다 — v3 산출물이 깨졌습니다")
            comp_of[idx] = gi
    missing = set(range(n_hi)) - set(comp_of.keys())
    if missing:
        raise ValueError(f"연결요소가 커버 못 한 HI 인덱스: {sorted(missing)}")
    return comp_of


def main() -> None:
    args = _parse_args()
    v3 = json.loads(Path(args.v3_groups_json).read_text(encoding="utf-8"))
    interaction = json.loads(Path(args.interaction_json).read_text(encoding="utf-8"))

    ref_seg = interaction["scenario_names"][0]
    ref_cols = get_hi_cols_for_seg(ref_seg)
    n_hi = len(ref_cols)
    suffix = f"_{ref_seg}"
    concepts_in_order = [c[: -len(suffix)] if c.endswith(suffix) else c for c in ref_cols]
    per_hi = interaction["per_hi"]

    # shared_hi_mask와 완전히 동일한 정의(phase1_trainer_v2.py의 --interaction-json 로더와
    # 100% 같은 기준 재사용 — 중복 구현이지만 한쪽만 import하기엔 순환 의존이라 로직만 맞춤).
    is_specific = [bool(per_hi.get(c, {"significant": False})["significant"]) for c in concepts_in_order]
    specific_idx = [i for i, s in enumerate(is_specific) if s]  # 오름차순 — _specific_idx와 동일 순서
    n_specific = len(specific_idx)
    print(f"[build] 특이 HI {n_specific}/{n_hi}개(공유 {n_hi - n_specific}개는 그룹 대상에서 제외)")

    n_scen = 0
    while f"seg_{n_scen}_groups" in v3:
        n_scen += 1

    out: dict = {
        "tag": args.tag,
        "source_v3_groups_json": str(args.v3_groups_json),
        "source_interaction_json": str(args.interaction_json),
        "n_hi": n_hi,
        "n_specific": n_specific,
        "specific_idx": specific_idx,
    }

    for s in range(n_scen):
        groups = v3[f"seg_{s}_groups"]
        attached = v3.get(f"seg_{s}_group_attached", [[] for _ in groups])
        comp_of = _build_components(groups, attached, n_hi)

        local_group_ids: list[int] = []
        remap: dict[int, int] = {}
        for idx in specific_idx:
            cid = comp_of[idx]
            if cid not in remap:
                remap[cid] = len(remap)
            local_group_ids.append(remap[cid])

        # 정합성 검증: 길이가 특이 개수와 정확히 같아야 SCRModel의 scen_gate_width와 맞는다.
        assert len(local_group_ids) == n_specific, (
            f"seg_{s}: group_ids 길이({len(local_group_ids)}) != n_specific({n_specific})"
        )
        n_groups_s = len(remap)
        n_singleton = sum(1 for g in remap.values() if local_group_ids.count(g) == 1)

        out[f"seg_{s}_seg_name"] = v3.get(f"seg_{s}_seg_name", f"seg_{s}")
        out[f"seg_{s}_specific_group_ids"] = local_group_ids
        out[f"seg_{s}_n_groups"] = n_groups_s
        print(f"[build] {out[f'seg_{s}_seg_name']}: 특이 {n_specific}개 -> 그룹 {n_groups_s}개 "
              f"(단독 그룹 {n_singleton}개, 최대 그룹 크기 "
              f"{max(local_group_ids.count(g) for g in set(local_group_ids))})")

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"specific_component_groups_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build] 저장: {out_path}")


if __name__ == "__main__":
    main()
