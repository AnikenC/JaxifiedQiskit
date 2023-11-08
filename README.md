# Jaxifying Qiskit Dynamics

This repository aims to use jax to accelerate qiskit dynamics, qiskit pulse, and qiskit runtime, in conjunction to produce incredibly fast hamiltonian simulations. This reduces simulations for one to three qubit systems by orders of magnitude, allowing for millisecond level simulation of statevector and density matrix evolutions.

NOTES:
Modified qiskit Drag source coded class to remove parameter constraints (avoids explicit tracer value checks during jit-compilation)

HOW:
To get faster, some tricks are:

1. Use a statevector as y0 when possible, avoids the need of Lindbladian Simulation
2. Truncate hamiltonian levels (ie use dim=3 for non-linearities, higher level aren't ideal for simulation speed)
3. Reduce rtol and atol whenever possible, fast values are such as rtol=1e-5, atol=1e-3 or even smaller
4. If only the final state is desired, simpliyg the t_eval linspace to [t0, t1] to reduce unnecessary calculations
5. Use jitted and vmapped simulations across large batches to reduce sim durations by orders of magnitude
6. Disable parameter checks upon Pulse instantiation (or remove the constraints from the source code) to allow for easy jitting
