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


def PauliToQuditMatrix(inp_str: str, qudit_dim_size: Optional[int] = 4):
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
