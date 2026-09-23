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
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import N_HI, RAW_CH, RAW_N, spec_from_qfrac
from models.cap_heads import build_cap_head

# 5_model/models/scr_model.py → repo root (train_scr.py/test_scr.py의 PROJECT_ROOT와 동일 계산)
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
        scen_group_ids: Optional[dict[int, list[int]]] = None,  # Phase 1: build_synergy_groups.py
            # 산출물 — {scenario_idx: [group_id per HI]}. 주어진 시나리오는 scen_gates가
            # GroupedHardConcreteGate로, 없는 시나리오는 기존 HardConcreteGate로 만들어진다.
        n_kernel_hi: int = 0,  # Phase 1: build_kernel_group_features.py 산출물 — 그룹당 RBF
            # 커널 융합 HI를 raw HI(x_hi)를 대체하지 않고 "추가"로 넣을 때의 폭. 0이면 기존과
            # 완전히 동일(커널 블록 없음). >0이면 시나리오별 scen_kernel_gates(N_HI와 별개
            # 폭 n_kernel_hi)가 추가로 생기고, cap_head 입력에 그 블록이 덧붙는다.
        shared_hi_mask: Optional[torch.Tensor] = None,  # Phase 1(v4): bool (N_HI,) —
            # test_hi_scenario_interaction.py 산출물 기반. True인 HI는 시나리오 무관 단일
            # shared_gate로, False인 HI는 기존처럼 시나리오별 scen_gates로 라우팅한다.
            # None(기본)이면 완전히 비활성 — 기존과 100% 동일 동작(전부 scen_gates).
            # scen_group_ids와 동시 사용 가능: shared_hi_mask가 있으면 scen_gate_width가
            # specific 폭으로 좁아지고, scen_group_ids[s]도 그 좁은 폭(len(specific_idx))
            # 기준 로컬 인덱스여야 한다.
        n_gate_groups: int | None = None,  # 2026-09-17: docs/260917_REPORT.md 안건2
            # "게이트 분리 대신 시나리오를 원샷 입력으로" 실험용 — scen_gates(+
            # scen_kernel_gates)의 폭을 n_scenarios 대신 이 값으로 줄이고, 각
            # scenario_id를 spec.scenario_to_dir_class(sid)[0](방향, 0/1)로 묶어서
            # 그 방향의 게이트로 라우팅한다. None(기본)이면 기존과 100% 동일
            # (게이트 폭 = n_scenarios). noscen(assign=none)과 게이트 파편화 프로필을
            # 맞추면서, 라벨은 position_bin(진짜 zone 정보)을 그대로 쓰는 조합에 사용.
        scenario_onehot: bool = False,  # 2026-09-17: 위와 짝을 이루는 옵션 — batch["level"]
            # (zone/latent_class, n_classes 범주)의 원-핫을 cap_head 입력에 direction/
            # cap_init처럼 그냥 이어붙인다. 게이트 라우팅과는 완전히 독립된 별도 경로 —
            # "시나리오 정보를 게이트로 나눠서 전달"이 아니라 "그냥 입력으로 알려주기"를
            # 테스트하기 위함. False(기본)면 기존과 100% 동일(head_in 안 바뀜).
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
            # 다름, 그냥 존재하지 않는 슬롯의 0-채움). n_gate_groups(방향 축소)와는 동시
            # 사용 불가(어느 시나리오의 K_s를 대표로 쓸지 불명확) — 동시 사용 시 assert.
            # None(기본)이면 기존과 100% 동일(n_kernel_hi 균일 폭).
        kernel_hi_costs: Optional[dict[int, list[float]]] = None,  # 2026-09-18(L0 비용
            # 가중치): {scenario_idx: [비용, ...]} — 길이는 kernel_hi_counts[s]와 동일해야
            # 한다(로컬 순서도 동일해야 함, build_kernel_group_features.py가 저장한 f["cost"]
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
            # (phase1_trainer_v2.py의 _apply_combined_redundancy_raw, 이제 삭제됨), 그
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

        # n_gate_groups: scen_gates(+scen_kernel_gates)의 실제 게이트 뱅크 크기를
        # n_scenarios 대신 이 값으로 줄이고, scenario_idx -> group_idx로 라우팅한다.
        # n_gate_groups=1: 전체 시나리오가 단일 공유 게이트 하나로(방향 구분도 없음) —
        # "웜스타트 후 분기"(docs/260917_REPORT.md, branch_scen_gates() 참고) Phase 1 전용.
        # n_gate_groups=2: 방향(충전/방전)별로만 묶음 — scenario_to_dir_class 기반.
        self.n_gate_groups = n_gate_groups
        if n_gate_groups is not None:
            if n_gate_groups == 1:
                _group_of = [0] * self.n_scenarios
            else:
                _group_of = [spec.scenario_to_dir_class(s)[0] for s in range(self.n_scenarios)]
                assert max(_group_of) + 1 <= n_gate_groups, (
                    f"n_gate_groups={n_gate_groups}인데 scenario_to_dir_class가 만든 그룹 "
                    f"인덱스 최대값이 {max(_group_of)} — n_gate_groups를 그룹 수 이상으로 주세요."
                )
            self.register_buffer("_gate_group_map",
                                  torch.tensor(_group_of, dtype=torch.long), persistent=False)
            _gate_bank_size = n_gate_groups
        else:
            self._gate_group_map = None
            _gate_bank_size = self.n_scenarios

        self.scenario_onehot = scenario_onehot

        if redundancy_mask is not None:
            assert redundancy_mask.shape == (self.n_scenarios, N_HI), (
                f"redundancy_mask shape {tuple(redundancy_mask.shape)}가 "
                f"(n_scenarios={self.n_scenarios}, N_HI={N_HI})와 다릅니다."
            )
            self.register_buffer("redundancy_mask", redundancy_mask.float(), persistent=False)
        else:
            self.redundancy_mask = None

        from models.hard_concrete import HardConcreteGate, GroupedHardConcreteGate

        if n_gate_groups is not None and scen_group_ids:
            raise ValueError(
                "n_gate_groups(게이트 뱅크 축소)와 scen_group_ids(전역 scenario_idx로 키된 "
                "GroupedHardConcreteGate)는 동시에 쓸 수 없습니다 — scen_group_ids의 키가 "
                "축소된 그룹 인덱스와 안 맞습니다."
            )

        # ----------------------------------------------------------------
        # Stage A — direction-aware probe gates
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Stage B — per-scenario gates (n_scenarios × N_HI), 단 shared_hi_mask가 있으면
        # 그 HI들은 이 폭에서 빠지고 아래 shared_gate가 대신 담당한다(v4).
        # ----------------------------------------------------------------
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
                    for s in range(_gate_bank_size)
                ])
            self._fixed_scen = False
        else:
            self.register_buffer("_scen_masks_buf", scen_masks.float())
            self.scen_gates = None
            self._fixed_scen = True

        # ----------------------------------------------------------------
        # Stage B' — 커널 융합 HI 블록(선택) — raw HI(scen_gates)를 대체하지 않고
        # 별도 폭(n_kernel_hi)의 독립 게이트로 "추가"한다. build_kernel_group_features.py가
        # 그룹당 1개씩 만든 RBF 커널 특징을 소비하는 용도(다중공선성/시너지 그룹 정보를
        # raw HI와 나란히 쓰고 싶을 때). n_kernel_hi=0이면 완전히 비활성(기존과 동일 동작).
        # ----------------------------------------------------------------
        if kernel_hi_counts is not None:
            assert n_gate_groups is None, (
                "kernel_hi_counts(시나리오별 커널 게이트 폭 축소)와 n_gate_groups(방향 축소)는 "
                "동시에 쓸 수 없습니다 — 그룹으로 묶인 여러 시나리오 중 어느 K_s를 대표로 "
                "쓸지가 불명확합니다."
            )
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
        else:
            self.kernel_hi_counts = None
            self.n_kernel_hi = n_kernel_hi
            if n_kernel_hi > 0:
                self.scen_kernel_gates = nn.ModuleList(
                    [HardConcreteGate(n_kernel_hi) for _ in range(_gate_bank_size)]
                )
            else:
                self.scen_kernel_gates = None

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

        # ----------------------------------------------------------------
        # Capacity head
        # input: probe_x (N_HI) || scen_x (N_HI) [|| kernel_x (n_kernel_hi)] [|| cnn_emb (3)]
        #        || direction (1) || cap_init (1)
        # = m active probe HIs + k active scen HIs [+ 커널 융합 HI] [+ raw V/I/t CNN 임베딩 3D]
        #   + 2 스칼라
        # Phase 1: 항상 MLP (model_cfg=None)
        # Phase 2: model_cfg["regression_model"] 에 따라
        #   mlp / transformer / i_transformer / resnet_tab / ft_transformer
        # ----------------------------------------------------------------
        self.cap_head = build_cap_head(model_cfg or {}, d_head=d_head, dropout=dropout,
                                        n_kernel_hi=self.n_kernel_hi,
                                        n_scen_onehot=(self.n_classes if scenario_onehot else 0))

        # ----------------------------------------------------------------
        # raw_cnn — 회귀 헤드용 원시 V/|I| 곡선 CNN 임베딩 (REGRESSION_UPGRADE.md §5/§8)
        # with_raw_cnn=False(기본) → 회귀 경로 완전히 기존과 동일(x_raw 무시).
        # with_raw_cnn=True:
        #   raw_cnn_pretrained_from 미지정 → RawCNN 랜덤 초기화, Phase2와 함께 학습 (방안 (b))
        #   raw_cnn_pretrained_from=<classifier clf_best.pt 경로> → 그 체크포인트의
        #     RawCNN 서브모듈("cnn.*")만 가중치 로드 후 얼림(requires_grad_(False)+eval 고정)
        #     — 사전 검증 (a): 분류기 CNN 재사용, Phase2 MSE 그래디언트가 CNN에 안 흐름.
        # ----------------------------------------------------------------
        _mcfg = model_cfg or {}
        self.with_raw_cnn = bool(_mcfg.get("with_raw_cnn", False))
        self._raw_cnn_frozen = False
        if self.with_raw_cnn:
            from models.raw_cnn import RawCNN
            self.raw_cnn = RawCNN()   # 출력 3D 고정: [h_scen,h_intensity,h_soh] (docs/260803_RESULTS.md §10)
            _pretrained_from = _mcfg.get("raw_cnn_pretrained_from")
            if _pretrained_from:
                _pf_path = Path(_pretrained_from)
                if not _pf_path.is_absolute():
                    _pf_path = _PROJECT_ROOT / _pf_path
                _ckpt = torch.load(_pf_path, map_location="cpu")
                _state = _ckpt["clf_state"] if isinstance(_ckpt, dict) and "clf_state" in _ckpt else _ckpt
                _cnn_state = {
                    k[len("cnn."):]: v for k, v in _state.items() if k.startswith("cnn.")
                }
                missing, unexpected = self.raw_cnn.load_state_dict(_cnn_state, strict=True)
                for p in self.raw_cnn.parameters():
                    p.requires_grad_(False)
                self.raw_cnn.eval()
                self._raw_cnn_frozen = True
                print(f"[scr_model] raw_cnn: frozen, loaded from {_pf_path}")
            else:
                print("[scr_model] raw_cnn: random init, trainable (Phase2와 함께 학습)")
        else:
            self.raw_cnn = None

        # ----------------------------------------------------------------
        # raw_flat — 방안1(REGRESSION_UPGRADE.md §2 방안1): raw V/|I| 곡선을 압축 없이
        # flatten(RAW_CH*RAW_N=96)해 그대로 concat. with_raw_cnn과 동시 사용 불가(택1).
        # HI는 이미 z-score(mean0/std1)인데 raw_v(~3-4V)/raw_i(~0-5A)는 스케일이 전혀
        # 다르므로, 문서 원안(단순 reshape concat)에 BatchNorm1d를 하나 더해 정규화한다
        # — RawCNN이 stem에서 BatchNorm1d로 채널 스케일을 흡수하는 것과 동등한 처리를
        # 주지 않으면 raw 블록이 gradient를 불공정하게 지배해 비교가 왜곡된다.
        # ----------------------------------------------------------------
        self.with_raw_flat = bool(_mcfg.get("with_raw_flat", False))
        if self.with_raw_cnn and self.with_raw_flat:
            raise ValueError("with_raw_cnn과 with_raw_flat을 동시에 켤 수 없습니다 (방안2 vs 방안1).")
        if self.with_raw_flat:
            self.raw_flat_norm = nn.BatchNorm1d(RAW_CH * RAW_N)
        else:
            self.raw_flat_norm = None

        # ----------------------------------------------------------------
        # probe_mlp — Phase 1 dual-objective CE head
        # probe_x (N_HI, mostly zeros) → n_classes logits
        # CE gradient flows through probe_mlp → probe_gate only
        # MSE gradient flows through cap_head → probe_gate + scen_gates
        # ----------------------------------------------------------------
        if with_probe_mlp:
            # 입력: probe_x (N_HI) + direction (1) → N_HI+1
            # direction 추가로 충/방전 간 sparsity 패턴 구분
            self.probe_mlp: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(N_HI + 1, d_probe),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_probe, d_probe // 2),
                nn.ReLU(),
                nn.Linear(d_probe // 2, self.n_classes),
            )
        else:
            self.probe_mlp = None

    # ------------------------------------------------------------------
    # train()/eval() 오버라이드 — 얼린 raw_cnn은 BatchNorm 통계도 절대 갱신되면 안 됨
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """부모 train(mode)를 호출한 뒤, raw_cnn이 얼려져 있으면 항상 eval()로 되돌린다.

        requires_grad_(False)는 그래디언트만 막을 뿐 BatchNorm의 러닝 통계
        갱신(forward 시 버퍼 업데이트, 그래디언트와 무관)은 막지 못한다 —
        model.train()이 재귀적으로 raw_cnn.training=True를 만들면 frozen CNN의
        BatchNorm이 Phase2 데이터 분포로 계속 오염된다. 그걸 막기 위한 오버라이드.
        """
        super().train(mode)
        if self._raw_cnn_frozen and self.raw_cnn is not None:
            self.raw_cnn.eval()
        return self

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
        """
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

        n_gate_groups가 설정된 경우(고정 마스크 scen_masks와는 동시에 안 씀 — Phase1 학습
        전용) scen_idx를 여기서 그룹 인덱스로 먼저 치환한다 — 아래 모든 분기가
        치환된 scen_idx만 보면 되도록.

        redundancy_mask(2026-09-18, 요구사항2를 raw HI에도 게이트 구조로 강제)가 있으면,
        위 세 분기 중 무엇을 타든 상관없이 마지막에 한 번만 적용한다 — **원본**(그룹
        치환 전) scen_idx로 인덱싱해야 한다(마스크는 진짜 시나리오 기준으로 만들어졌지,
        n_gate_groups로 묶인 그룹 기준이 아니므로). False인 자리는 masked/z_out 둘 다
        무조건 0이 된다 — log_alpha가 뭐라고 하든 "고른 결과 자체가 0"이라 커널 쪽
        (kernel_hi_counts로 슬롯을 아예 없앤 것)과 동일한 강도의 보장이다."""
        orig_scen_idx = scen_idx
        if self._gate_group_map is not None:
            scen_idx = self._gate_group_map[scen_idx]
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
            row_mask = self.redundancy_mask[orig_scen_idx]  # (B, N_HI)
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
        각 행은 자기 시나리오 own 슬롯 [0:K_s)만 실값이고 나머지는 phase1_trainer_v2.py가
        이미 0-패딩해둔 상태)를 시나리오별로 [0:K_s) 구간만 잘라 그 폭의 게이트에 넣고,
        결과를 다시 max_k 폭으로 되돌린다(뒤쪽은 항상 0)."""
        if self._gate_group_map is not None:
            scen_idx = self._gate_group_map[scen_idx]
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

        # Capacity head: probe_x + scen_x [+ 커널 융합 HI] [+ raw CNN 임베딩 | raw flat]
        #                + direction + cap_init
        feat_parts = [probe_x, scen_x]
        if self.scen_kernel_gates is not None:
            x_kernel = batch["x_kernel"]                     # (B, n_kernel_hi), 이미 정규화됨
            kernel_x, kernel_z = self._apply_scen_kernel_gate(x_kernel, scen_idx)
            feat_parts.append(kernel_x)
        else:
            kernel_z = None
        if self.raw_cnn is not None:
            if self._raw_cnn_frozen:
                with torch.no_grad():
                    cnn_emb = self.raw_cnn(batch["x_raw"])       # (B, 3)=[h_scen,h_intensity,h_soh] — 그래디언트 차단
            else:
                cnn_emb = self.raw_cnn(batch["x_raw"])           # (B, 3) — Phase2와 함께 학습
            feat_parts.append(cnn_emb)
        elif self.with_raw_flat:
            x_raw = batch["x_raw"]                                # (B, RAW_CH, RAW_N)
            raw_flat = self.raw_flat_norm(x_raw.reshape(x_raw.size(0), -1))  # (B, 96)
            feat_parts.append(raw_flat)
        if self.scenario_onehot:
            # 2026-09-17 안건2: 게이트 라우팅과 무관하게 "진짜 zone/level"을 그냥
            # 입력으로 이어붙인다 — batch["level"]은 segment_dataset.py가 만든 0/1/2
            # ground-truth(CE 라벨과 동일 소스), n_classes 폭 원-핫.
            level_onehot = F.one_hot(batch["level"], num_classes=self.n_classes).to(x.dtype)
            feat_parts.append(level_onehot)
        feat_parts += [direction.unsqueeze(1), batch["cap_init"].unsqueeze(1)]
        feat = torch.cat(feat_parts, dim=1)                  # (B, 2*N_HI+2) 또는 (B, 2*N_HI+3+2)/(B, 2*N_HI+96+2)
        cap_pred = self.cap_head(feat)                       # (B,)

        # CE head: [probe_x || direction] → class logits (Phase 1 dual-objective only)
        if self.probe_mlp is not None:
            probe_x_dir = torch.cat([probe_x, direction.unsqueeze(1)], dim=1)
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

    def branch_scen_gates(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """웜스타트 후 분기(docs/260917_REPORT.md 안건2 다음 단계, phase1_trainer_v2.py의
        --warmstart-branch-epoch T 전용): n_gate_groups=1로 학습된 단일 공유 scen_gates[0]
        (모든 시나리오 데이터로 학습됨, L0/이산화 압력 없음)을 n_scenarios개의 독립
        HardConcreteGate로 복제해 분기한다. 각 새 게이트의 log_alpha는 공유 게이트의
        log_alpha를 detach().clone()해 초기화하고(그 시점 이후로는 완전히 독립적으로,
        자기 시나리오 데이터만으로 학습됨), 호출 이후 이 모델은 n_gate_groups=None인
        평범한 6-게이트 SCRModel과 완전히 동일하게 동작한다(_gate_group_map도 해제).

        Returns (old_params, new_params) — 호출자(트레이너)가 옵티마이저 상태를 old는
        제거하고(momentum 폐기) new는 새로 추가(모멘텀 0부터 재시작)하는 데 쓴다
        (docs/260917_REPORT.md: "Adam 모멘텀 재시작" 결정).

        scen_group_ids(그룹 계층 게이팅)와는 아직 함께 쓸 수 없다(단일 공유 게이트가
        일반 HardConcreteGate여야 한다는 가정)."""
        assert self.n_gate_groups == 1, (
            "branch_scen_gates()는 n_gate_groups=1(전체 시나리오 공유 게이트 1개)로 만든 "
            f"모델에서만 호출할 수 있습니다 (현재 n_gate_groups={self.n_gate_groups})."
        )
        assert self.shared_gate is None, "branch_scen_gates()는 shared_hi_mask(v4)와 함께 쓸 수 없습니다."

        from models.hard_concrete import HardConcreteGate

        old_gate = self.scen_gates[0]
        old_params = list(old_gate.parameters())
        width = old_gate.log_alpha.numel()
        device = old_gate.log_alpha.device

        new_gates = nn.ModuleList([HardConcreteGate(width) for _ in range(self.n_scenarios)]).to(device)
        with torch.no_grad():
            for g in new_gates:
                g.log_alpha.copy_(old_gate.log_alpha)

        self.scen_gates = new_gates
        self.n_gate_groups = None
        self._gate_group_map = None
        new_params = list(self.scen_gates.parameters())
        return old_params, new_params
