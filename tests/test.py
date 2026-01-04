from shades.stabilizer import StabilizerState
import numpy as np

s = StabilizerState(np.array([[1,-1]]))
ham = [[1, 'X']]
spawned_states = s.apply_hamiltonian(ham)
for s in spawned_states:
     print(s.generator_matrix, spawned_states[s])