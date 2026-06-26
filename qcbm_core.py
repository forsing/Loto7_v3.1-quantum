"""QCBM jezgro — rotacije, entanglement, Born mašina, MMD gubitak (iz lottery_easy_Quantum, 7/39)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumRotationLayer(nn.Module):
    """Parametarski rotacioni sloj (Z-Y-Z / X-Y / samo Z)."""

    def __init__(self, n_qubits: int, rotation_type: str = "xyz"):
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation_type = rotation_type
        if rotation_type == "xyz":
            self.params = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
        elif rotation_type == "xy":
            self.params = nn.Parameter(torch.randn(n_qubits, 2) * 0.1)
        else:
            self.params = nn.Parameter(torch.randn(n_qubits, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        n_qubits = int(math.log2(dim)) if dim > 0 else self.n_qubits
        result = x
        if self.rotation_type == "xyz":
            for q in range(min(n_qubits, self.n_qubits)):
                rz1, ry, rz2 = self.params[q, 0], self.params[q, 1], self.params[q, 2]
                result = self._apply_zyz_rotation(result, q, n_qubits, rz1, ry, rz2)
        elif self.rotation_type == "xy":
            for q in range(min(n_qubits, self.n_qubits)):
                rx, ry = self.params[q, 0], self.params[q, 1]
                result = self._apply_xy_rotation(result, q, n_qubits, rx, ry)
        else:
            for q in range(min(n_qubits, self.n_qubits)):
                rz = self.params[q, 0]
                result = self._apply_z_rotation(result, q, n_qubits, rz)
        return result

    def _apply_ry(self, state: torch.Tensor, qubit: int, n_qubits: int, angle: torch.Tensor) -> torch.Tensor:
        cos_a = torch.cos(angle / 2)
        sin_a = torch.sin(angle / 2)
        dim = state.shape[-1]
        step = 1 << (n_qubits - 1 - qubit)
        result = torch.zeros_like(state)
        for base in range(0, dim, 2 * step):
            for i in range(step):
                idx0 = base + i
                idx1 = base + i + step
                a0 = state[..., idx0]
                a1 = state[..., idx1]
                result[..., idx0] = cos_a * a0 - sin_a * a1
                result[..., idx1] = sin_a * a0 + cos_a * a1
        return result

    def _apply_z_rotation(self, state: torch.Tensor, qubit: int, n_qubits: int, angle: torch.Tensor) -> torch.Tensor:
        cos_a = torch.cos(angle / 2)
        dim = state.shape[-1]
        indices = torch.arange(dim, device=state.device)
        bits = (indices >> (n_qubits - 1 - qubit)) & 1
        scale = torch.where(
            bits == 0,
            torch.ones_like(indices, dtype=torch.float32),
            torch.full_like(indices, dtype=torch.float32, fill_value=cos_a.item()),
        )
        return state * scale.float().to(state.device)

    def _apply_zyz_rotation(
        self, state: torch.Tensor, qubit: int, n_qubits: int, rz1, ry, rz2
    ) -> torch.Tensor:
        s = self._apply_z_rotation(state, qubit, n_qubits, rz2)
        s = self._apply_ry(s, qubit, n_qubits, ry)
        s = self._apply_z_rotation(s, qubit, n_qubits, rz1)
        return s

    def _apply_xy_rotation(self, state: torch.Tensor, qubit: int, n_qubits: int, rx, ry) -> torch.Tensor:
        s = self._apply_ry(state, qubit, n_qubits, ry)
        s = self._apply_z_rotation(s, qubit, n_qubits, rx)
        return s


class QuantumEntanglementLayer(nn.Module):
    """Parametarski CNOT / entanglement sloj."""

    def __init__(self, n_qubits: int, mode: str = "circular"):
        super().__init__()
        self.n_qubits = n_qubits
        self.mode = mode
        self.entangle_params = nn.Parameter(torch.randn(n_qubits, n_qubits) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        n_qubits = int(math.log2(dim)) if dim > 0 else self.n_qubits
        result = x
        for c, t in self._get_entangle_pairs(n_qubits):
            strength = torch.sigmoid(
                self.entangle_params[c % self.n_qubits, t % self.n_qubits]
            )
            result = self._apply_cnot(result, c, t, n_qubits, strength)
        return result

    def _get_entangle_pairs(self, n_qubits: int) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        if self.mode == "linear":
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        elif self.mode == "circular":
            pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        elif self.mode == "full":
            pairs = [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
        elif self.mode == "block":
            half = n_qubits // 2
            for i in range(half):
                for j in range(i + 1, half):
                    pairs.append((i, j))
            for i in range(half, n_qubits):
                for j in range(i + 1, n_qubits):
                    pairs.append((i, j))
        elif self.mode == "correlation":
            pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
            pairs += [(i, (i + 2) % n_qubits) for i in range(n_qubits)]
        return pairs

    def _apply_cnot(
        self, state: torch.Tensor, control: int, target: int, n_qubits: int, strength: torch.Tensor
    ) -> torch.Tensor:
        dim = state.shape[-1]
        c_step = 1 << (n_qubits - 1 - control)
        t_step = 1 << (n_qubits - 1 - target)
        result = state.clone()
        for base in range(dim):
            if base & c_step == 0:
                continue
            partner = base ^ t_step
            if partner < base:
                continue
            a0 = state[..., base]
            a1 = state[..., partner]
            result[..., base] = (1 - strength) * a0 + strength * a1
            result[..., partner] = strength * a0 + (1 - strength) * a1
        return result


class QCBM(nn.Module):
    """Kvantni Born stroj — uči raspodelu brojeva 1..N."""

    def __init__(self, num_range: int, config: dict):
        super().__init__()
        self.num_range = num_range
        self.n_qubits = max(1, math.ceil(math.log2(max(num_range, 2))))
        self.state_dim = 2**self.n_qubits
        self.config = config
        qcbm_cfg = config["qcbm"]
        self.n_layers = qcbm_cfg["n_layers"]

        init_type = qcbm_cfg["initial_state"]
        if init_type == "uniform":
            self.register_buffer(
                "initial_state", torch.ones(self.state_dim) / math.sqrt(self.state_dim)
            )
        elif init_type == "random":
            init = torch.randn(self.state_dim)
            init = init / torch.norm(init)
            self.register_buffer("initial_state", init)
        else:
            init = torch.zeros(self.state_dim)
            init[0] = 1.0
            self.register_buffer("initial_state", init)

        self.rotation_layers = nn.ModuleList()
        self.entangle_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.rotation_layers.append(
                QuantumRotationLayer(self.n_qubits, qcbm_cfg["rotation_type"])
            )
            self.entangle_layers.append(
                QuantumEntanglementLayer(self.n_qubits, qcbm_cfg["entanglement_mode"])
            )

        self.output_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
        )

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        state = self.initial_state.unsqueeze(0).expand(batch_size, -1)
        for layer in range(self.n_layers):
            state = self.rotation_layers[layer](state)
            state = self.entangle_layers[layer](state)
        state = self.output_layer(state)
        probs = state**2
        probs = probs[:, : self.num_range]
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        return probs


class QuantumLossFunctions:
    """MMD i pomoćne funkcije gubitka."""

    @staticmethod
    def mmd_loss(generated: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        mmd_cfg = config["mmd"]
        kernel = mmd_cfg["kernel"]
        sigmas = mmd_cfg["sigma"]
        if kernel == "rbf":
            total_mmd = torch.tensor(0.0, device=generated.device)
            for sigma in sigmas:
                gamma = 1.0 / (2 * sigma**2)
                xx = QuantumLossFunctions._pairwise_sq_dist(generated)
                yy = QuantumLossFunctions._pairwise_sq_dist(target)
                xy = QuantumLossFunctions._cross_sq_dist(generated, target)
                kxx = torch.exp(-gamma * xx).mean()
                kyy = torch.exp(-gamma * yy).mean()
                kxy = torch.exp(-gamma * xy).mean()
                total_mmd = total_mmd + kxx + kyy - 2 * kxy
            return total_mmd / len(sigmas)
        if kernel == "linear":
            kxx = generated @ generated.T
            kyy = target @ target.T
            kxy = generated @ target.T
            return kxx.mean() + kyy.mean() - 2 * kxy.mean()
        kxx = (generated @ generated.T + 1) ** 2
        kyy = (target @ target.T + 1) ** 2
        kxy = (generated @ target.T + 1) ** 2
        return kxx.mean() + kyy.mean() - 2 * kxy.mean()

    @staticmethod
    def _pairwise_sq_dist(X: torch.Tensor) -> torch.Tensor:
        dot = X @ X.T
        norm_sq = torch.diag(dot).unsqueeze(1)
        dist = norm_sq + norm_sq.T - 2 * dot
        return torch.clamp(dist, min=0)

    @staticmethod
    def _cross_sq_dist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        xx = (X**2).sum(dim=-1, keepdim=True)
        yy = (Y**2).sum(dim=-1, keepdim=True)
        dist = xx + yy.T - 2 * X @ Y.T
        return torch.clamp(dist, min=0)
