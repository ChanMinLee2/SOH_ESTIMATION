"""
SCR (Scenario-Conditioned Routing) model.

Stage A — direction-aware probe gate (dual objective)
  1. Direction-aware HardConcreteGate:
       charge_probe_gate    (N_HI) — charging segments
       discharge_probe_gate (N_HI) — discharging segments
     Phase 1 (with_probe_mlp=True):
       probe_gate receives gradients from BOTH MSE (regression) and CE (classification)
       → selects HIs useful for both tasks simultaneously
     Phase 2 / inference: probe_gate frozen from Phase 1 JSON (regression utility only)

Stage B — scenario-conditioned regression
  2. Per-scenario HardConcreteGate (n_scenarios × N_HI) selects k HIs per scenario
     MSE gradient only — regression-specialised subset per scenario

  3. Capacity head: [probe_x || scen_x || direction || cap_init] → SOH ratio

At inference JSON masks may replace the L0 gates (fixed binary vectors).

2026-10-04: SCRModel.__init__(225줄)을 단일 책임 원칙에 따라 4개 빌더 메서드로
분리했다(동작 변화 없는 순수 구조 정리) — _build_probe_gates(Stage A) ->
_build_scen_gates(Stage B) -> _build_kernel_gates(Stage B') -> (cap_head는 그대로
인라인) -> _build_probe_mlp. 같은 라운드에서 self.raw_cnn/with_raw_cnn/
_raw_cnn_frozen(2026-09-25부터 영구 False/None 고정 데드 스텁 — 의존하는
models/raw_cnn.py 자체가 repo에 없어 한 번도 실행된 적 없는 코드였음)을 완전히
삭제하고 forward()의 대응 분기도 제거했다. model_lib/tools/visualize_results.py는
`getattr(model, "raw_cnn", None)`로 이미 방어적으로 읽고 있어(속성이 아예 없어도
None 반환, 기존과 동일 동작) 수정 불필요 — 실제로 안 건드림.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import N_HI, spec_from_qfrac
from models.cap_heads import build_cap_head
from models.hard_concrete import HardConcreteGate, GroupedHardConcreteGate

# model_lib/models/scr_model.py → repo root (train_scr.py/test_scr.py의 PROJECT_ROOT와 동일 계산)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SCRModel(nn.Module):
    """
    Args:
        d_probe          : hidden dim for Stage A MLP
        d_head           : hidden dim for capacity head
        dropout          : dropout rate
        charge_probe_mask   : fixed bool (N_HI,) for charging probe (Phase 2 / test)
        discharge_probe_mask: fixed bool (N_HI,) for discharging probe (Phase 2 / test)
        scen_masks       : fixed bool (N_SEGS, N_HI) for scenario gates (Phase 2 / test)
    """

    def __init__(
        self,
        d_probe: int = 64,
        d_head: int = 128,
        dropout: float = 0.1,
        charge_probe_mask: Optional[torch.Tensor] = None,    # (N_HI,) bool
        discharge_probe_mask: Optional[torch.Tensor] = None, # (N_HI,) bool
        scen_masks: Optional[torch.Tensor] = None,           # (n_scenarios, N_HI) bool
        model_cfg: Optional[dict] = None,  # Phase 2 전용: regression_model 선택
        spec=None,   # ScenarioSpec | None  (None → qfrac default)
        with_probe_mlp: bool = False,  # Phase 1 dual-objective: probe_gate에 CE 그래디언트 추가
        scen_group_ids: Optional[dict[int, list[int]]] = None,  # Phase 1: synergy.py
            # 산출물 — {scenario_idx: [group_id per HI]}. 주어진 시나리오는 scen_gates가
            # GroupedHardConcreteGate로, 없는 시나리오는 기존 HardConcreteGate로 만들어진다.
        n_kernel_hi: int = 0,  # Phase 1: kernel.py 산출물 — 그룹당 RBF
            # 커널 융합 HI를 raw HI(x_hi)를 대체하지 않고 "추가"로 넣을 때의 폭. 0이면 기존과
            # 완전히 동일(커널 블록 없음). >0이면 시나리오별 scen_kernel_gates(N_HI와 별개
            # 폭 n_kernel_hi)가 추가로 생기고, cap_head 입력에 그 블록이 덧붙는다.
        shared_hi_mask: Optional[torch.Tensor] = None,  # Phase 1(v4): bool (N_HI,) —
            # interaction.py 산출물 기반. True인 HI는 시나리오 무관 단일
            # shared_gate로, False인 HI는 기존처럼 시나리오별 scen_gates로 라우팅한다.
            # None(기본)이면 완전히 비활성 — 기존과 100% 동일 동작(전부 scen_gates).
            # scen_group_ids와 동시 사용 가능: shared_hi_mask가 있으면 scen_gate_width가
            # specific 폭으로 좁아지고, scen_group_ids[s]도 그 좁은 폭(len(specific_idx))
            # 기준 로컬 인덱스여야 한다.
        kernel_hi_counts: Optional[list[int]] = None,  # 2026-09-18(v4 로직 수정, 요구사항1을
            # "데이터에서 0으로 죽이기"가 아니라 게이트 구조 자체로 강제): 길이 n_scenarios,
            # 시나리오별 실제 own 커널 HI 개수(K_s). 주어지면 scen_kernel_gates[s]의 폭이
            # n_kernel_hi(전체 K, 모든 시나리오 공통) 대신 K_s로 좁아진다 — 그 시나리오
            # 게이트에는 애초에 다른 시나리오 커널을 위한 슬롯 자체가 없다(이전엔 슬롯은
            # 있되 입력이 상수라 게이트가 '고를 수는 있지만 의미 없는' 상태였음 — 실측
            # 결과 실제로 다수 선택되는 게 확인돼 이 방식으로 교체). cap_head 입력 폭은
            # max(kernel_hi_counts)로 고정하고(배치 텐서는 폭이 균일해야 하므로), 각
            # 시나리오는 자기 폭 K_s만큼만 앞쪽에 채우고 나머지는 0-패딩한다(다른 시나리오
            # 폭이 더 넓어서 생기는 여유 슬롯일 뿐 — 이전의 "다른 시나리오 것을 빌려옴"과는
            # 다름, 그냥 존재하지 않는 슬롯의 0-채움).
            # None(기본)이면 기존과 100% 동일(n_kernel_hi 균일 폭).
        kernel_hi_costs: Optional[dict[int, list[float]]] = None,  # 2026-09-18(L0 비용
            # 가중치): {scenario_idx: [비용, ...]} — 길이는 kernel_hi_counts[s]와 동일해야
            # 한다(로컬 순서도 동일해야 함, kernel.py가 저장한 f["cost"]
            # = 그 커널을 만든 멤버 raw HI들의 카테고리 비용(stat/diff/lfp/morph) 평균).
            # scr_loss.py의 커널 L0 페널티가 gate_prob() 합을 균일 비용(1.0) 대신 이
            # 가중치로 곱해서 합산하는 데 쓴다 — raw HI가 cost_vec(카테고리별)로 하는 것과
            # 동일한 취지. kernel_hi_counts 없이는 의미가 없다(assert). None(기본)이면
            # 커널 L0 페널티가 기존처럼 균일 비용 1.0을 쓴다(100% 하위호환).
        redundancy_mask: Optional[torch.Tensor] = None,  # 2026-09-18(v4 로직 수정, 요구사항2를
            # raw HI에도 게이트 구조로 강제): bool (n_scenarios, N_HI) — build_kernel_group_
            # features.py의 3차 결합(raw+kernel) 다중공선성 배제 결과(*_combined_redundancy.json)
            # 중 raw 쪽. True=이 시나리오에서 이 raw HI 허용, False=배제. 커널 HI는 K_s로 게이트
            # 자체를 좁혀서 슬롯을 아예 없앴지만, raw HI는 64개 카탈로그를 모든 시나리오가
            # 공유하는 구조(다른 여러 스크립트가 이 가정에 의존)라 폭을 줄이는 대신, 학습된
            # scen_gates 출력(masked/z 둘 다)에 이 마스크를 곱해 False인 자리는 log_alpha가
            # 뭐라고 하든 항상 0으로 강제한다 — "고를 수는 있지만 기여는 0"이 아니라 "고른
            # 결과 자체가 무조건 0"이라 커널 쪽과 동일한 강도의 보장이 된다.
            # 2026-09-21: 예전엔 입력 레벨에서도 nan_mask를 0으로 이중 강제했는데
            # (train.py의 _apply_combined_redundancy_raw, 이제 삭제됨), 그
            # nan_mask가 probe_x(분류기 입력)에도 공유돼 분류기 정확도가 붕괴하는 버그였다
            # — 이 게이트 레벨 마스크 하나만으로 scen_x 쪽 정확성은 이미 충분히 보장되므로
            # 입력 레벨 이중 마스킹은 제거했다. None(기본)이면 기존과 100% 동일.
    ):
        super().__init__()
        self.d_probe = d_probe
        self.d_head = d_head

        # Resolve spec (stored as plain Python attr; save separately as JSON)
        if spec is None:
            spec = spec_from_qfrac()
        self.spec = spec
        self.n_scenarios = spec.n_scenarios
        self.n_classes   = spec.n_classes

        _gate_bank_size = self.n_scenarios

        # 방향(충전/방전)별 후보 시나리오 개수 — 분류기(Stage A)가 "그 방향 안의
        # 후보 시나리오 전부"를 보게 할 때(2026-10-07, probe_kernel_gates/
        # probe_redundancy_mask) 두 방향의 폭이 같아야 probe_mlp 입력 차원이
        # 고정된다. 정식 6-시나리오 레시피는 3/3, no_scen(assign="none")은 1/1이라
        # 항상 대칭 — 비대칭 축이 생기면 여기서 바로 에러를 내 조용한 오동작을 막는다.
        n_ch_scen = len(spec.charge_scenario_ids)
        n_dis_scen = len(spec.discharge_scenario_ids)
        assert n_ch_scen == n_dis_scen, (
            f"charge/discharge 시나리오 개수가 다릅니다({n_ch_scen} vs {n_dis_scen}) — "
            "probe_kernel_gates/probe_redundancy_mask는 양방향 폭이 같다고 가정합니다."
        )
        self._n_dir_scenarios = n_ch_scen

        if redundancy_mask is not None:
            assert redundancy_mask.shape == (self.n_scenarios, N_HI), (
                f"redundancy_mask shape {tuple(redundancy_mask.shape)}가 "
                f"(n_scenarios={self.n_scenarios}, N_HI={N_HI})와 다릅니다."
            )
            self.register_buffer("redundancy_mask", redundancy_mask.float(), persistent=False)
            # 2026-10-07: 분류기(Stage A)용 합집합 마스크 — redundancy_mask는
            # (n_scenarios, N_HI)라 시나리오를 알아야 행을 고를 수 있는데, 분류기는
            # direction만 알고 시나리오는 모른다(그걸 맞추는 게 분류기의 일). 그래서
            # 그 방향에 속한 시나리오들의 마스크를 "합집합"(하나라도 허용하면 허용)으로
            # 합쳐 (2, N_HI) 텐서를 만든다 — 사용자 확정(교집합 대신 합집합: 정보 보존
            # 우선, §결론 및 향후 방향 1번).
            ch_mask  = redundancy_mask[spec.charge_scenario_ids].any(dim=0)
            dis_mask = redundancy_mask[spec.discharge_scenario_ids].any(dim=0)
            probe_redundancy_mask = torch.stack([ch_mask, dis_mask], dim=0)  # (2, N_HI)
            self.register_buffer("probe_redundancy_mask", probe_redundancy_mask.float(), persistent=False)
        else:
            self.redundancy_mask = None
            self.probe_redundancy_mask = None

        self._build_probe_gates(charge_probe_mask, discharge_probe_mask)  # Stage A
        self._build_scen_gates(scen_masks, scen_group_ids, shared_hi_mask, _gate_bank_size)  # Stage B
        self._build_kernel_gates(n_kernel_hi, kernel_hi_counts, kernel_hi_costs, _gate_bank_size)  # Stage B'

        # Capacity head — input: probe_x (N_HI) || scen_x (N_HI) [|| kernel_x (n_kernel_hi)]
        # || direction (1) || cap_init (1). Phase 1: 항상 MLP(model_cfg=None). Phase 2:
        # model_cfg["regression_model"]에 따라 mlp/transformer/i_transformer/resnet_tab/
        # ft_transformer.
        self.cap_head = build_cap_head(model_cfg or {}, d_head=d_head, dropout=dropout,
                                        n_kernel_hi=self.n_kernel_hi)

        self._build_probe_mlp(with_probe_mlp, d_probe, dropout)  # Phase 1 dual-objective CE head

    def _build_probe_gates(
        self, charge_probe_mask: Optional[torch.Tensor], discharge_probe_mask: Optional[torch.Tensor],
    ) -> None:
        """Stage A — direction-aware probe gates. 두 마스크가 모두 주어지면(Phase 2/test)
        학습 가능한 게이트 대신 고정 buffer를 쓰고, 아니면(Phase 1) HardConcreteGate를 만든다."""
        fixed_probe = (charge_probe_mask is not None and
                       discharge_probe_mask is not None)
        self._fixed_probe = fixed_probe

        if not fixed_probe:
            self.charge_probe_gate    = HardConcreteGate(N_HI)
            self.discharge_probe_gate = HardConcreteGate(N_HI)
        else:
            self.charge_probe_gate    = None
            self.discharge_probe_gate = None
            self.register_buffer("_charge_probe_mask_buf",    charge_probe_mask.float())
            self.register_buffer("_discharge_probe_mask_buf", discharge_probe_mask.float())

    def _build_scen_gates(
        self, scen_masks: Optional[torch.Tensor], scen_group_ids: Optional[dict[int, list[int]]],
        shared_hi_mask: Optional[torch.Tensor], gate_bank_size: int,
    ) -> None:
        """Stage B — per-scenario gates (n_scenarios × N_HI), 단 shared_hi_mask가 있으면
        그 HI들은 이 폭에서 빠지고 shared_gate가 대신 담당한다(v4)."""
        self.shared_gate = None
        if shared_hi_mask is not None:
            shared_hi_mask = shared_hi_mask.bool()
            shared_idx = torch.nonzero(shared_hi_mask, as_tuple=False).flatten()
            specific_idx = torch.nonzero(~shared_hi_mask, as_tuple=False).flatten()
            self.register_buffer("_shared_idx", shared_idx)
            self.register_buffer("_specific_idx", specific_idx)
            if len(shared_idx) > 0:
                self.shared_gate = HardConcreteGate(len(shared_idx))
            scen_gate_width = len(specific_idx)
        else:
            scen_gate_width = N_HI

        if scen_masks is None:
            scen_group_ids = scen_group_ids or {}
            self.scen_gates = nn.ModuleList([
                GroupedHardConcreteGate(scen_gate_width, scen_group_ids[s]) if s in scen_group_ids
                    else HardConcreteGate(scen_gate_width)
                    for s in range(gate_bank_size)
                ])
            self._fixed_scen = False
        else:
            self.register_buffer("_scen_masks_buf", scen_masks.float())
            self.scen_gates = None
            self._fixed_scen = True

    def _build_kernel_gates(
        self, n_kernel_hi: int, kernel_hi_counts: Optional[list[int]],
        kernel_hi_costs: Optional[dict[int, list[float]]], gate_bank_size: int,
    ) -> None:
        """Stage B' — 커널 융합 HI 블록(선택) — raw HI(scen_gates)를 대체하지 않고 별도
        폭(n_kernel_hi)의 독립 게이트로 "추가"한다. kernel.py가 그룹당 1개씩 만든 RBF
        커널 특징을 소비하는 용도. n_kernel_hi=0이면 완전히 비활성(기존과 동일 동작).
        kernel_hi_costs가 주어지면 scr_loss.py의 커널 L0 페널티 비용 가중치로 쓸
        self.kernel_hi_costs를 검증 후 채운다.

        2026-10-07: 분류기(Stage A)용 probe_kernel_gates도 여기서 함께 만든다 —
        scen_kernel_gates와 완전히 같은 모양(시나리오별 K_s 폭)이지만 독립된
        파라미터다. scen_kernel_gates는 "진짜 시나리오" 행만 보고
        (_apply_gate_list가 scen_idx로 라우팅), probe_kernel_gates는 forward()에서
        "그 방향의 후보 시나리오 전부"에 적용된다(진짜 시나리오를 모르는 분류기가
        회귀와 동일한 커널 HI+다중공선성 배제 로직을 쓰되, 라우팅 기준만 scen_idx
        대신 direction으로 바뀐 것) — 입력 행 선택 기준이 달라 파라미터를 공유하지
        않는다."""
        if kernel_hi_counts is not None:
            assert len(kernel_hi_counts) == self.n_scenarios, (
                f"kernel_hi_counts 길이({len(kernel_hi_counts)})가 n_scenarios"
                f"({self.n_scenarios})와 다릅니다."
            )
            self.kernel_hi_counts = list(kernel_hi_counts)
            self.n_kernel_hi = max(kernel_hi_counts) if kernel_hi_counts else 0
            # K_s=0(그 시나리오가 own 커널을 하나도 못 만든 경우, 드묾)인 시나리오도
            # None 대신 폭 1의 더미 게이트로 만든다 — 어차피 그 슬롯은 항상 0-패딩만
            # 받으므로 ModuleList/저장/포화도 집계 코드에서 None 분기를 따로 둘 필요가 없다.
            self.scen_kernel_gates = nn.ModuleList(
                [HardConcreteGate(max(k, 1)) for k in kernel_hi_counts]
            )
            self.probe_kernel_gates = nn.ModuleList(
                [HardConcreteGate(max(k, 1)) for k in kernel_hi_counts]
            )
        else:
            self.kernel_hi_counts = None
            self.n_kernel_hi = n_kernel_hi
            if n_kernel_hi > 0:
                self.scen_kernel_gates = nn.ModuleList(
                    [HardConcreteGate(n_kernel_hi) for _ in range(gate_bank_size)]
                )
                self.probe_kernel_gates = nn.ModuleList(
                    [HardConcreteGate(n_kernel_hi) for _ in range(gate_bank_size)]
                )
            else:
                self.scen_kernel_gates = None
                self.probe_kernel_gates = None

        if kernel_hi_costs is not None:
            assert self.kernel_hi_counts is not None, (
                "kernel_hi_costs는 kernel_hi_counts가 주어졌을 때만 의미가 있습니다."
            )
            costs_list: list[list[float]] = []
            for s in range(self.n_scenarios):
                k_s = self.kernel_hi_counts[s]
                c = list(kernel_hi_costs.get(s, []))
                if k_s == 0:
                    c = [1.0]  # 더미 폭-1 게이트(K_s=0) — 항상 0-패딩만 받으므로 값 무관
                assert len(c) == max(k_s, 1), (
                    f"kernel_hi_costs[{s}] 길이({len(c)})가 kernel_hi_counts[{s}]"
                    f"({max(k_s, 1)})와 다릅니다."
                )
                costs_list.append(c)
            self.kernel_hi_costs = costs_list
        else:
            self.kernel_hi_costs = None

    def _build_probe_mlp(self, with_probe_mlp: bool, d_probe: int, dropout: float) -> None:
        """probe_mlp — Phase 1 dual-objective CE head. probe_x(N_HI, 대부분 0)+direction(1)
        [+ 커널 후보 블록] -> n_classes logits. CE 그래디언트는 probe_mlp를 거쳐
        probe_gate(+probe_kernel_gates)에만 흐르고, MSE 그래디언트는 cap_head를 거쳐
        probe_gate+scen_gates(+scen_kernel_gates)에 흐른다. with_probe_mlp=False면
        None(기존 동작).

        2026-10-07: probe_kernel_gates가 있으면(§_build_kernel_gates) 입력 폭에
        그 방향의 후보 시나리오 수(self._n_dir_scenarios) × 커널 폭(self.n_kernel_hi)
        만큼 추가한다 — 없으면(기존 레시피, 커널 HI 자체가 없는 run) N_HI+1 그대로라
        100% 하위호환."""
        if with_probe_mlp:
            # 입력: probe_x (N_HI) + direction (1) [+ 커널 후보 (n_dir*n_kernel_hi)]
            # direction 추가로 충/방전 간 sparsity 패턴 구분
            kernel_extra = (self._n_dir_scenarios * self.n_kernel_hi
                             if self.probe_kernel_gates is not None else 0)
            self.probe_mlp: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(N_HI + 1 + kernel_extra, d_probe),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_probe, d_probe // 2),
                nn.ReLU(),
                nn.Linear(d_probe // 2, self.n_classes),
            )
        else:
            self.probe_mlp = None

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _apply_probe_gate(
        self, x: torch.Tensor, direction: torch.Tensor, scen_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Routes each sample to its direction-specific probe gate.
        x         : (B, N_HI)
        direction : (B,)  +1.0=charge, -1.0=discharge
        scen_idx   : (B,)  0-5
        Returns (probe_x, probe_z): both (B, N_HI)

        probe_redundancy_mask(2026-10-07, 분류기에도 회귀와 동일한 다중공선성
        배제를 반영)이 있으면 마지막에 곱한다 — redundancy_mask와 동일한 "강제
        0" 패턴(§_apply_scen_gate 참고), 다만 행 선택 기준이 scen_idx 대신
        direction(그 방향 3개 시나리오의 합집합 — 하나라도 허용하면 허용)."""
        B = x.size(0)
        probe_x = torch.zeros_like(x)
        probe_z = torch.zeros_like(x)

        ch_sel  = (direction > 0)   # charging samples
        dis_sel = (direction <= 0)  # discharging samples

        if self._fixed_probe:
            if ch_sel.any():
                m = self._charge_probe_mask_buf          # (N_HI,)
                n = int(ch_sel.sum().item())
                probe_x[ch_sel] = x[ch_sel] * m
                probe_z[ch_sel] = m.unsqueeze(0).expand(n, -1)
            if dis_sel.any():
                m = self._discharge_probe_mask_buf
                n = int(dis_sel.sum().item())
                probe_x[dis_sel] = x[dis_sel] * m
                probe_z[dis_sel] = m.unsqueeze(0).expand(n, -1)
        else:
            if ch_sel.any():
                mx, zz = self.charge_probe_gate(x[ch_sel])
                probe_x[ch_sel] = mx
                probe_z[ch_sel] = zz
            if dis_sel.any():
                mx, zz = self.discharge_probe_gate(x[dis_sel])
                probe_x[dis_sel] = mx
                probe_z[dis_sel] = zz

        if self.probe_redundancy_mask is not None:
            dir_idx = (direction <= 0).long()           # 0=charge, 1=discharge
            row_mask = self.probe_redundancy_mask[dir_idx]  # (B, N_HI)
            probe_x = probe_x * row_mask
            probe_z = probe_z * row_mask

        return probe_x, probe_z

    @staticmethod
    def _apply_gate_list(
        gates, x: torch.Tensor, scen_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """시나리오별 게이트 컬렉션(scen_gates 또는 scen_kernel_gates) 공통 라우팅 로직.
        각 세그먼트를 자기 시나리오(scen_idx)에 해당하는 gates[s]로만 통과시킨다.
        x: (B, width) — width는 게이트 종류에 따라 다름(raw HI면 N_HI, 커널 HI면 n_kernel_hi).
        gates: nn.ModuleList(독립 HardConcreteGate/GroupedHardConcreteGate).
        Returns (masked_x, z): 둘 다 x와 같은 shape."""
        masked = torch.zeros_like(x)
        z_out  = torch.zeros_like(x)
        for s, gate in enumerate(gates):
            sel = (scen_idx == s)
            if sel.any():
                mx, zz = gate(x[sel])
                masked[sel] = mx
                z_out[sel]  = zz
        return masked, z_out

    def _apply_scen_gate(
        self, x: torch.Tensor, scen_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """raw HI(scen_gates, 폭 N_HI) 전용. Phase2는 고정 마스크(scen_masks)를 쓸 수 있어
        그 경우만 별도 분기 — 학습 가능한 게이트일 때는 _apply_gate_list로 위임.

        shared_gate가 있으면(v4) N_HI 폭을 shared_idx/specific_idx로 쪼개서, shared 부분은
        시나리오 무관 단일 shared_gate로(scen_idx 라우팅 없음 — 전체 배치에 동일 적용),
        specific 부분만 기존처럼 scen_gates[s]로 라우팅한 뒤 원래 컬럼 위치로 재조립한다.
        재조립하므로 반환 shape/컬럼 순서는 shared_gate 유무와 무관하게 항상 (B,N_HI) 그대로라
        cap_head 등 하위 코드는 변경이 필요 없다.

        redundancy_mask(2026-09-18, 요구사항2를 raw HI에도 게이트 구조로 강제)가 있으면,
        위 세 분기 중 무엇을 타든 상관없이 마지막에 한 번만 적용한다. False인 자리는
        masked/z_out 둘 다 무조건 0이 된다 — log_alpha가 뭐라고 하든 "고른 결과 자체가
        0"이라 커널 쪽(kernel_hi_counts로 슬롯을 아예 없앤 것)과 동일한 강도의 보장이다."""
        if self.shared_gate is not None:
            masked = torch.zeros_like(x)
            z_out = torch.zeros_like(x)
            x_shared = x[:, self._shared_idx]
            m_shared, z_shared = self.shared_gate(x_shared)
            masked[:, self._shared_idx] = m_shared
            z_out[:, self._shared_idx] = z_shared
            if len(self._specific_idx) > 0:
                x_specific = x[:, self._specific_idx]
                m_specific, z_specific = self._apply_gate_list(self.scen_gates, x_specific, scen_idx)
                masked[:, self._specific_idx] = m_specific
                z_out[:, self._specific_idx] = z_specific
        elif self._fixed_scen:
            masked = torch.zeros_like(x)
            z_out  = torch.zeros_like(x)
            for s in range(self.n_scenarios):
                sel = (scen_idx == s)
                if sel.any():
                    m = self._scen_masks_buf[s]
                    n_sel = int(sel.sum().item())
                    masked[sel] = x[sel] * m
                    z_out[sel]  = m.unsqueeze(0).expand(n_sel, -1)
        else:
            masked, z_out = self._apply_gate_list(self.scen_gates, x, scen_idx)

        if self.redundancy_mask is not None:
            row_mask = self.redundancy_mask[scen_idx]  # (B, N_HI)
            masked = masked * row_mask
            z_out = z_out * row_mask
        return masked, z_out

    def _apply_scen_kernel_gate(
        self, x: torch.Tensor, scen_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """커널 융합 HI(scen_kernel_gates) 전용 — 고정 마스크 개념이 아직 없으므로
        (Phase1 전용) 항상 학습 가능한 게이트를 적용한다.

        kernel_hi_counts가 없으면(기존 동작) 모든 시나리오 게이트가 동일 폭 n_kernel_hi라
        _apply_gate_list로 바로 위임한다. kernel_hi_counts가 있으면(2026-09-18, 요구사항1을
        게이트 구조로 강제) 시나리오마다 게이트 폭이 K_s로 다르므로, x(폭 max_k=n_kernel_hi,
        각 행은 자기 시나리오 own 슬롯 [0:K_s)만 실값이고 나머지는 train.py가
        이미 0-패딩해둔 상태)를 시나리오별로 [0:K_s) 구간만 잘라 그 폭의 게이트에 넣고,
        결과를 다시 max_k 폭으로 되돌린다(뒤쪽은 항상 0)."""
        if self.kernel_hi_counts is None:
            return self._apply_gate_list(self.scen_kernel_gates, x, scen_idx)

        masked = torch.zeros_like(x)
        z_out = torch.zeros_like(x)
        for s, gate in enumerate(self.scen_kernel_gates):
            sel = (scen_idx == s)
            if not sel.any():
                continue
            k_s = self.kernel_hi_counts[s]
            if k_s == 0:
                continue  # 이 시나리오는 own 커널이 0개 -- 더미 게이트(폭1)엔 애초에 실값이 없음
            mx, zz = gate(x[sel][:, :k_s])
            masked[sel, :k_s] = mx
            z_out[sel, :k_s] = zz
        return masked, z_out

    def _apply_probe_kernel_gate(
        self, x_kernel_probe: torch.Tensor, direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """2026-10-07 신설 — 분류기(Stage A)용 커널 후보 블록.

        scen_kernel_gates는 "진짜 시나리오" 행만 골라 그 시나리오 게이트에
        통과시키지만, 분류기는 시나리오를 모르는 입장이라(그걸 맞추는 게 분류기의
        일) 그 대신 "이 세그먼트의 direction에 속한 후보 시나리오 전부"를
        각자의 own 커널 게이트(probe_kernel_gates[s] — scen_kernel_gates[s]와
        같은 모양, 독립 파라미터)에 통과시켜 이어붙인다. train.py가
        x_kernel_probe((B, n_scenarios, max_k) — 모든 시나리오에 대해 미리
        predict해둔 값, ds.x_kernel처럼 진짜 scen_idx로 고른 게 아니라 전부
        보존)를 batch에 실어준다.

        x_kernel_probe : (B, n_scenarios, max_k)
        direction      : (B,)  +1.0=charge, -1.0=discharge
        Returns (masked, z): 둘 다 (B, n_dir_scenarios * max_k) — direction과
        무관하게 폭이 고정(probe_mlp가 고정 크기 입력을 받아야 하므로).
        """
        B = x_kernel_probe.size(0)
        max_k = self.n_kernel_hi
        n_dir = self._n_dir_scenarios
        masked = torch.zeros(B, n_dir, max_k, device=x_kernel_probe.device, dtype=x_kernel_probe.dtype)
        z_out = torch.zeros_like(masked)

        ch_sel  = (direction > 0)
        dis_sel = (direction <= 0)
        for dir_sel, scen_ids in (
            (ch_sel, self.spec.charge_scenario_ids),
            (dis_sel, self.spec.discharge_scenario_ids),
        ):
            if not dir_sel.any():
                continue
            for slot, s in enumerate(scen_ids):
                k_s = self.kernel_hi_counts[s] if self.kernel_hi_counts is not None else max_k
                if k_s == 0:
                    continue  # 그 시나리오는 own 커널이 0개 — 더미 게이트(폭1)엔 애초에 실값이 없음
                x_s = x_kernel_probe[dir_sel, s, :k_s]
                mx, zz = self.probe_kernel_gates[s](x_s)
                masked[dir_sel, slot, :k_s] = mx
                z_out[dir_sel, slot, :k_s] = zz
        return masked.reshape(B, n_dir * max_k), z_out.reshape(B, n_dir * max_k)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        batch keys: x_hi (B,N_HI), nan_mask (B,N_HI), direction (B,),
                    scen_idx (B,), cap_init (B,)

        Returns:
          cap_pred     : (B,)      SOH ratio prediction
          level_logits : (B,n_cls) class logits (real when probe_mlp active, else zeros)
          probe_x      : (B,N_HI) direction-masked probe features (for Phase 3 classifier)
          probe_z      : (B,N_HI) gate activation values
          scen_z       : (B,N_HI) scenario gate activation values
        """
        x         = batch["x_hi"]                           # (B, N_HI)
        nan_mask  = batch["nan_mask"]                       # (B, N_HI)
        direction = batch["direction"]                      # (B,)
        scen_idx   = batch["scen_idx"]                        # (B,)

        x = x * nan_mask  # NaN positions → 0

        # Stage A: direction-aware probe gate
        # MSE gradient → probe_gate (regression signal)
        # CE gradient  → probe_gate via probe_mlp (classification signal, Phase 1 only)
        probe_x, probe_z = self._apply_probe_gate(x, direction, scen_idx)

        # Stage B: scenario-conditioned gate (MSE gradient only)
        scen_x, scen_z = self._apply_scen_gate(x, scen_idx) # (B, N_HI)

        # Capacity head: probe_x + scen_x [+ 커널 융합 HI] + direction + cap_init
        feat_parts = [probe_x, scen_x]
        if self.scen_kernel_gates is not None:
            x_kernel = batch["x_kernel"]                     # (B, n_kernel_hi), 이미 정규화됨
            kernel_x, kernel_z = self._apply_scen_kernel_gate(x_kernel, scen_idx)
            feat_parts.append(kernel_x)
        else:
            kernel_z = None
        feat_parts += [direction.unsqueeze(1), batch["cap_init"].unsqueeze(1)]
        feat = torch.cat(feat_parts, dim=1)                  # (B, 2*N_HI+2) 또는 (B, 2*N_HI+n_kernel_hi+2)
        cap_pred = self.cap_head(feat)                       # (B,)

        # CE head: [probe_x || direction [|| 커널 후보]] → class logits
        # (Phase 1 dual-objective only). 2026-10-07: probe_kernel_gates가 있으면
        # 그 방향의 후보 시나리오 전부의 커널 HI를 이어붙인다(§_apply_probe_kernel_gate).
        probe_kernel_z = None
        if self.probe_mlp is not None:
            probe_mlp_parts = [probe_x, direction.unsqueeze(1)]
            if self.probe_kernel_gates is not None:
                x_kernel_probe = batch["x_kernel_probe"]   # (B, n_scenarios, max_k)
                probe_kernel_x, probe_kernel_z = self._apply_probe_kernel_gate(x_kernel_probe, direction)
                probe_mlp_parts.append(probe_kernel_x)
            probe_x_dir = torch.cat(probe_mlp_parts, dim=1)
            level_logits = self.probe_mlp(probe_x_dir)
        else:
            level_logits = torch.zeros(
                x.size(0), self.n_classes, dtype=x.dtype, device=x.device
            )

        out = {
            "cap_pred":     cap_pred,
            "level_logits": level_logits,
            "probe_x":      probe_x,
            "probe_z":      probe_z,
            "scen_z":       scen_z,
        }
        if kernel_z is not None:
            out["kernel_z"] = kernel_z
        if probe_kernel_z is not None:
            out["probe_kernel_z"] = probe_kernel_z
        return out

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(batch)

    @torch.no_grad()
    def get_probe_x(
        self, x_hi: torch.Tensor, direction: torch.Tensor, scen_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Direction-aware probe masking for external use (e.g., classifier inference).
        Returns probe_x (B, N_HI) — same shape as x_hi but only m active positions.
        """
        probe_x, _ = self._apply_probe_gate(x_hi, direction, scen_idx)
        return probe_x

    @torch.no_grad()
    def get_probe_kernel_x(
        self, x_kernel_probe: torch.Tensor, direction: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """get_probe_x의 커널 버전(2026-10-07) — scr_evaluator.py가 forward()
        전체를 안 거치고 분류기 입력만 수동으로 재구성할 때(hard/soft 라우팅 평가)
        쓴다. probe_kernel_gates가 없으면(커널 HI 자체가 없는 run) None."""
        if self.probe_kernel_gates is None:
            return None
        x_k, _ = self._apply_probe_kernel_gate(x_kernel_probe, direction)
        return x_k

    @torch.no_grad()
    def get_selected_probe_his(self) -> dict[str, list[int]]:
        """Returns {"charge": [active_hi_indices], "discharge": [...]}."""
        if self._fixed_probe:
            return {
                "charge":    self._charge_probe_mask_buf.nonzero(as_tuple=False).squeeze(1).tolist(),
                "discharge": self._discharge_probe_mask_buf.nonzero(as_tuple=False).squeeze(1).tolist(),
            }
        return {
            "charge":    self.charge_probe_gate.active_indices(),
            "discharge": self.discharge_probe_gate.active_indices(),
        }

    @torch.no_grad()
    def get_selected_scen_his(self) -> dict[int, list[int]]:
        """Returns {scen_idx: [active_hi_indices]}."""
        if self._fixed_scen:
            return {
                s: self._scen_masks_buf[s].nonzero(as_tuple=False).squeeze(1).tolist()
                for s in range(self.n_scenarios)
            }
        return {s: gate.active_indices() for s, gate in enumerate(self.scen_gates)}
