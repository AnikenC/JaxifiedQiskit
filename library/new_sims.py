import numpy as np
import qiskit
from qiskit import pulse

from qiskit_dynamics import Solver, DynamicsBackend
from qiskit_dynamics.pulse import InstructionToSignals

import jax.numpy as jnp
from jax import jit, vmap, block_until_ready

import chex

from typing import Optional, Union

from library.utils import PauliToQuditOperator


class JaxedDynamicsBackend:
    def __init__(
        self,
    ):
        super().__init__()


class JaxedSolver:
    def __init__(
        self,
        schedule_func,
        solver,
        dt,
        carrier_freqs,
        ham_chans,
        ham_ops,
        t_span,
        rtol,
        atol,
    ):
        super().__init__()
        self.schedule_func = schedule_func
        self.solver = solver
        self.dt = dt
        self.carrier_freqs = carrier_freqs
        self.ham_chans = ham_chans
        self.ham_ops = ham_ops
        self.t_span = t_span
        self.rtol = rtol
        self.atol = atol
        self.fast_batched_sim = jit(vmap(self.run_sim))

    def run_sim(self, y0, obs, params):
        sched = self.schedule_func(params)

        converter = InstructionToSignals(
            self.dt, carriers=self.carrier_freqs, channels=self.ham_chans
        )

        signals = converter.get_signals(sched)

        results = self.solver.solve(
            t_span=self.t_span,
            y0=y0 / jnp.linalg.norm(y0),
            t_eval=self.t_span,
            signals=signals,
            rtol=self.rtol,
            atol=self.atol,
            convert_results=False,
            method="jax_odeint",
        )

        state_vec = results.y.data[-1]
        state_vec = state_vec / jnp.linalg.norm(state_vec)
        new_vec = obs @ state_vec
        probs_vec = jnp.abs(new_vec) ** 2
        probs_vec = jnp.clip(probs_vec, a_min=0.0, a_max=1.0)

        # Shots instead of probabilities

        return probs_vec

    def estimate2(self, batch_y0, batch_params, batch_obs_str):
        batch_obs = jnp.zeros(
            (batch_y0.shape[0], batch_y0.shape[1], batch_y0.shape[1]),
            dtype=jnp.complex64,
        )
        num_qubits = len(batch_obs_str[0])
        for i, b_str in enumerate(batch_obs_str):
            batch_obs = batch_obs.at[i].set(
                PauliToQuditOperator(b_str, int(batch_y0.shape[1] ** (1 / num_qubits)))
            )
        return self.fast_batched_sim(batch_y0, batch_obs, batch_params)
