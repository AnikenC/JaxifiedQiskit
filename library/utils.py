import numpy as np
import qiskit
from qiskit import pulse

from qiskit_dynamics import Solver, DynamicsBackend
from qiskit_dynamics.pulse import InstructionToSignals

import jax.numpy as jnp
from jax import jit, vmap, block_until_ready

import chex

from typing import Optional, Union

base_gates_dict = {
    "I": jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    "X": jnp.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": jnp.array([[0.0, -1.0j], [1.0j, 0.0]]),
    "Z": jnp.array([[1.0, 0.0], [0.0, -1.0]]),
}


def PauliToQuditOperator(inp_str: str, qudit_dim_size: Optional[int] = 4):
    """
    This function operates very similarly to SparsePauliOp from Qiskit, except this can produce
    arbitrary dimension qudit operators that are the equivalent of the Qubit Operators desired.

    This functionality is useful for qudit simulations of standard qubit workflows like state preparation
    and choosing measurement observables, without losing any information from the simulation.

    All operators produced remain as unitaries.
    """
    word_list = list(inp_str)
    qudit_op_list = []
    for word in word_list:
        qubit_op = base_gates_dict[word]
        qud_op = np.identity(qudit_dim_size, dtype=np.complex64)
        qud_op[:2, :2] = qubit_op
        qudit_op_list.append(qud_op)
    complete_op = qudit_op_list[0]
    for i in range(1, len(qudit_op_list)):
        complete_op = np.kron(complete_op, qudit_op_list[i])
    return complete_op


class TwoQuditHamiltonian:
    """
    Custom Two Coupled Qudit Hamiltonian for Qiskit Dynamics Testing.

    Important Attributes include:\n
    dt\n
    solver\n
    ham_ops\n
    ham_chans\n
    chan_freqs
    """

    def __init__(
        self,
        qudit_dim: int,
        dt: Optional[float] = 1 / 4.5e9,
    ):
        super().__init__()

        if not isinstance(dt, float):
            raise ValueError("dt needs to be a float")
        if not isinstance(qudit_dim, int):
            raise ValueError("qudit_dim needs to be an int")

        self.dt = dt
        self.dim = qudit_dim
        self.solver = self.make_solver()

    def make_solver(self):
        v0 = 4.86e9
        anharm0 = -0.32e9
        r0 = 0.22e9

        v1 = 4.97e9
        anharm1 = -0.32e9
        r1 = 0.26e9

        J = 0.002e9

        a = np.diag(np.sqrt(np.arange(1, self.dim)), 1)
        adag = np.diag(np.sqrt(np.arange(1, self.dim)), -1)
        N = np.diag(np.arange(self.dim))

        ident = np.eye(self.dim, dtype=complex)
        full_ident = np.eye(self.dim**2, dtype=complex)

        N0 = np.kron(ident, N)
        N1 = np.kron(N, ident)

        a0 = np.kron(ident, a)
        a1 = np.kron(a, ident)

        a0dag = np.kron(ident, adag)
        a1dag = np.kron(adag, ident)

        static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
        static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)

        static_ham_full = (
            static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
        )

        drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
        drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)

        self.ham_ops = [drive_op0, drive_op1, drive_op0, drive_op1]
        self.ham_chans = ["d0", "d1", "u0", "u1"]
        self.chan_freqs = {"d0": v0, "d1": v1, "u0": v1, "u1": v0}

        return Solver(
            static_hamiltonian=static_ham_full,
            hamiltonian_operators=self.ham_ops,
            rotating_frame=static_ham_full,
            hamiltonian_channels=self.ham_chans,
            channel_carrier_freqs=self.chan_freqs,
            dt=self.dt,
        )
