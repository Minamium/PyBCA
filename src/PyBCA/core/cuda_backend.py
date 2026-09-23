"""Optional nvcc-built CUDA kernels, independent of the PyTorch C++ ABI.

Build artifacts live in a content-addressed cache. No CUDA compiler or GPU is
needed to import PyBCA or to run the reference/Torch implementations.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch


def load_kernels(device):
    nvcc = os.environ.get("PYBCA_NVCC") or shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("execution_mode='cuda' needs nvcc on PATH (rokko: module load cuda/11.8.0 gcc/11.4.0)")
    source = Path(__file__).with_name("kernels.cu")
    major, minor = torch.cuda.get_device_capability(device)
    if major < 7:
        raise ValueError("CUDA candidate kernels require compute capability >= 7.0")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    digest = hashlib.sha256(source.read_bytes() + version.encode() + f"{major}{minor}".encode()).hexdigest()[:24]
    cache = Path(os.environ.get("PYBCA_CUDA_CACHE", Path.home()/".cache"/"pybca"))
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/f"kernels-{digest}.so"
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as td:
            output = Path(td)/"kernels.so"
            command = [nvcc, "--shared", "--cudart=static", "-Xcompiler=-fPIC", "-O3", "-std=c++14",
                       f"-arch=sm_{major}{minor}", str(source), "-o", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"CUDA kernel compilation failed:\n{result.stdout}\n{result.stderr}")
            os.replace(output, library)
    lib = ctypes.CDLL(str(library))
    lib.pybca_launch.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_float, ctypes.c_void_p]
    lib.pybca_launch.restype = ctypes.c_int
    return lib


class CudaPlan:
    def __init__(self, sim, capacity):
        if not str(sim.device).startswith("cuda"):
            raise ValueError("execution_mode='cuda' requires a CUDA device")
        self.sim = sim
        self.t, _, self.h, self.w = sim.TCHW.shape
        self.n = len(sim.rule_ids)
        if self.t > 65535 or self.t*self.h*self.w >= 2**31 or self.n >= 2**24:
            raise ValueError("CUDA dimensions exceed supported indexing limits")
        self.capacity = min(int(capacity), self.h*self.w)
        if self.t*self.n*self.capacity >= 2**31:
            raise ValueError("CUDA candidate buffer exceeds supported indexing limits")
        self.lib = load_kernels(sim.device)
        kw = dict(device=sim.device, dtype=torch.int32)
        self.candidates = torch.empty((self.t, self.n, self.capacity), **kw)
        self.counts = torch.zeros((self.t, self.n), **kw)
        self.rule_counts = torch.zeros_like(self.counts)
        self.order = torch.empty_like(self.counts)
        self.accepted = torch.empty((self.t, self.h, self.w), device=sim.device, dtype=torch.bool)
        self.keys = torch.zeros((self.t, 2), **kw)
        self.probs = sim.rule_probs_tensor
        lut = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        if sim.state_conversions is not None:
            for old, new in sim.state_conversions:
                lut[int(old)+128] = int(new)
        self.lut = torch.as_tensor(lut, device=sim.device)
        self.event_data = torch.empty((0, 7), dtype=torch.int64, device=sim.device)
        self.event_probs = torch.empty((0,), dtype=torch.float32, device=sim.device)
        self.event_hits = torch.empty((self.t, 0), dtype=torch.bool, device=sim.device)

    def set_keys(self, keys):
        self.keys = torch.as_tensor(keys.view(np.int32).copy(), device=self.sim.device)

    def launch(self, op, global_prob=1., independent=True):
        s = self.sim
        tensors = [s.TCHW, s.rule_arrays_tensor, self.probs, self.keys,
                   s.TNHW_boolMask, self.candidates, self.counts, s.TCHW_applied,
                   self.accepted, self.order, self.rule_counts]
        args = [x.data_ptr() for x in tensors]
        args += [self.t, self.n, self.h, self.w, self.capacity, s._current_step,
                 int(independent), self.n if self.probs.ndim == 2 else 0, int(s.record_rule_history)]
        args += [self.event_data.data_ptr(), self.event_probs.data_ptr(), self.event_hits.data_ptr(),
                 self.event_hits.shape[1], self.lut.data_ptr()]
        params = (ctypes.c_int64 * len(args))(*args)
        with torch.cuda.device(s.device):
            error = self.lib.pybca_launch(op, params, float(global_prob), torch.cuda.current_stream(s.device).cuda_stream)
        if error:
            raise RuntimeError(f"PyBCA CUDA launch failed, runtime error {error}, operation {op}")

    def update(self, global_prob, rng_mode):
        s = self.sim
        independent = rng_mode == "independent"
        self.probs = torch.as_tensor(s.rule_probs_tensor, dtype=torch.float32, device=s.device).contiguous()
        if independent and tuple(self.probs.shape) not in {(self.n,), (self.t, self.n)}:
            raise ValueError("Independent rule probabilities must be [N] or [T,N]")
        self.counts.zero_()
        self.rule_counts.zero_()
        self.launch(0, 1. if global_prob is None else global_prob, independent)
        if not independent:
            if global_prob is not None and global_prob != 1:
                s.TNHW_boolMask = s._global_prob_gate(global_prob)
            # Legacy RNG consumption, including the all-zero/all-one fast paths.
            if not bool(torch.all(s.rule_probs_tensor == 1)):
                s.TNHW_boolMask = s._rule_prob_gate()
            self.launch(1)
            perm = torch.randperm(self.n, generator=s.rng, device=s.device)
            self.order.copy_(perm[None])
        self.launch(2, independent=independent)

    def prepare_events(self, info):
        e = len(self.sim.spatial_event_names or [])
        rows = np.zeros((e, 7), dtype=np.int64)
        rows[:, 0] = -1
        probs = np.zeros(e, dtype=np.float32)
        for i, src, value, dst, new_value, prob, start, end in info:
            rows[i] = [src, value, dst, new_value, i, start, end]
            probs[i] = np.clip(prob, 0, 1)
        self.event_data = torch.as_tensor(rows, device=self.sim.device)
        self.event_probs = torch.as_tensor(probs, device=self.sim.device)
        self.event_hits = torch.zeros((self.t, e), dtype=torch.bool, device=self.sim.device)

    def events(self):
        self.launch(3)
        return self.event_hits

    def state_gates(self):
        self.launch(4)
