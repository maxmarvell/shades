import copy
import numpy as np
import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import stim
 
# Multiplication table for Pauli group with
# I = 0 X = 1 Y = 2 Z = 3
# [4-7] = -[0-3]
# [8-11] = i[0-3]
# [12-15] = -i[0-3]
pauli_multiplication_matrix = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [1, 0, 11, 14, 5, 4, 15, 10, 9, 8, 7, 2, 13, 12, 3, 6],
            [2, 15, 0, 9, 6, 11, 4, 13, 10, 3, 8, 5, 14, 7, 12, 1],
            [3, 10, 13, 0, 7, 14, 9, 4, 11, 6, 1, 8, 15, 2, 5, 12],
            [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
            [5, 4, 15, 10, 1, 0, 11, 14, 13, 12, 3, 6, 9, 8, 7, 2],
            [6, 11, 4, 13, 2, 15, 0, 9, 14, 7, 12, 1, 10, 3, 8, 5],
            [7, 14, 9, 4, 3, 10, 13, 0, 15, 2, 5, 12, 11, 6, 1, 8],
            [8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 0, 1, 2, 3],
            [9, 8, 7, 2, 13, 12, 3, 6, 5, 4, 15, 10, 1, 0, 11, 14],
            [10, 3, 8, 5, 14, 7, 12, 1, 6, 11, 4, 13, 2, 15, 0, 9],
            [11, 6, 1, 8, 15, 2, 5, 12, 7, 14, 9, 4, 3, 10, 13, 0],
            [12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
            [13, 12, 3, 6, 9, 8, 7, 2, 1, 0, 11, 14, 5, 4, 15, 10],
            [14, 7, 12, 1, 10, 3, 8, 5, 2, 15, 0, 9, 6, 11, 4, 13],
            [15, 2, 5, 12, 11, 6, 1, 8, 3, 10, 13, 0, 7, 14, 9, 4]],
            dtype = np.int8)

def get_2q_conjugation(gate):
    '''Get the results of conjugating all two-qubit Pauli strings by a two-qubit gate.
    IN:
       gate: 4x4 matrix representation of gate action
    OUT:
       gate effects: dictionary containing 2-qubit Pauli strings as keys and the results
        of their conjugation by gate as values. Uses the numerical convention above.
    '''
    i = np.array([[1,0],[0,1]])
    x = np.array([[0,1],[1,0]])
    y = np.array([[0,-1j],[1j,0]])
    z = np.array([[1,0],[0,-1]])

    paulis = {0:i, 1:x, 2:y, 3:z}

    gate_effects = {}
    pps = {}
    for c, cv in paulis.items():
         for d, dv in paulis.items():
             pps[(c,d)] = np.kron(cv, dv)
    for a, av in paulis.items():
        for b, bv in paulis.items():
            new = gate.dot(np.kron(av,bv).dot(gate.T))
            for c in paulis.keys():
                for d in paulis.keys():
                    if np.array_equal(new, pps[(c,d)]):
                        gate_effects[(a, b)] = (c, d)
                    if np.array_equal(new, -pps[(c,d)]):
                        gate_effects[(a, b)] = (c+4, d)
    return gate_effects

cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
cnot_transform = get_2q_conjugation(cnot)
cphase = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
cphase_transform = get_2q_conjugation(cphase)

class StabilizerState:
    '''Class encoding a Stabiliser State'''

    def __init__(self, generator_matrix, find_basis_state=True):
        '''Initialise StabilizerState.
        IN:
            generator_matrix: the generator matrix for the stabilizer state.
            find_basis_state: if True, find and store a basis state included in the stabilizer state.
        '''
        self.nrows = len(generator_matrix[:,0])
        self.generator_matrix, self.m = self.canonicalize(generator_matrix)
        if find_basis_state:
            self.basis_state, self.basis_state_overlap = self.find_basis_state()

    def __eq__(self, other):
        if not isinstance(other, StabilizerState):
            return NotImplemented
        return (self.generator_matrix == other.generator_matrix).all()

    def __hash__(self):
        return hash(tuple([tuple(row) for row in self.generator_matrix]))
    
    @classmethod
    def from_stim_tableau(cls, tableau: stim.Tableau):
        n = len(tableau)
        generator_matrix = np.zeros((n , n+1), dtype=np.int8)

        for i in range(n):
            pauli_string = tableau.z_output(i)
            
            sign = pauli_string.sign
            if sign == 1:
                generator_matrix[i, -1] = 1
            elif sign == -1:
                generator_matrix[i, -1] = -1
            else:
                raise RuntimeError("Pauli String should not have complex sign!")
            
            for j in range(n):
                pauli = pauli_string[j]  # Returns 0 (I), 1 (X), 2 (Y), 3 (Z)
                generator_matrix[i, j] = pauli
        
        return cls(generator_matrix, find_basis_state=True)


    @staticmethod
    def find_xy_row_index(matrix, nrows, j):
        '''Find the first row in a generator matrix that contains an X or Y Pauli
        in a particular position.
        IN:
            matrix: the generator matrix
            nrows: the number of rows
            j: the position of interest
        OUT:
            row: first row with X/Y in position j
        '''
        for row in range(nrows):
            if matrix[row][j] in (1,2):
                return row
        return None

    @staticmethod
    def find_z_row_index(matrix, nrows, j):
        '''Find the first row in a generator matrix that contains an Z Pauli
        in a particular position.
        IN:
            matrix: the generator matrix
            nrows: the number of rows
            j: the position of interest
        OUT:
            row: first row with Z in position j
        '''
        for row in range(nrows):
            if matrix[row][j] == 3:
                return row
        return None

    @staticmethod
    def find_last_z_row_index(matrix, nrows, j):
        '''Find the last row in a generator matrix that contains an Z Pauli
        in a particular position.
        IN:
            matrix: the generator matrix
            nrows: the number of rows
            j: the position of interest
        OUT:
            row: first row with Z in position j
        '''
        for row in range(nrows-1, -1, -1):
            if matrix[row][j] == 3:
                return row
        return None

    @staticmethod
    def rowmult(row1, row2):
        '''Multiply two rows of a generator matrix.
        IN:
            row1, row2: rows to be multiplied.
        OUT:
            row2: resulting row after multiplication.
        '''
        sign = 1+0j
        for i in range(len(row1)-1):
            row2[i] = pauli_multiplication_matrix[int(row1[i]), int(row2[i])]
            if 4 <= row2[i] < 8:
                #Pauli phase = -1
                sign *= -1
                row2[i] -= np.int8(4)
            elif 8 <= row2[i] < 12:
                #Pauli phase = i
                sign *=1j
                row2[i] -= np.int8(8)
            elif 12 <= row2[i]:
                #Pauli phase = -i
                sign *=-1j
                row2[i] -= np.int8(12)
        row2[-1] *= row1[-1]
        row2[-1] *= sign
        return row2

    def canonicalize(self, generator_matrix):
        '''Rewrite a generator matrix in canonical form.
        IN:
            self: StabilizerState
            generator_matrix: generator matrix
        OUT:
            generator_matrix: canonical upper echelon form of the generator.
            nondiagonal_rows: number of rows containing X/Y Paulis.
        '''
        generator_matrix = copy.copy(generator_matrix)
        nrows = self.nrows
        i = 0
        for j in range(nrows):
            k = self.find_xy_row_index(generator_matrix[i:], nrows-i, j)
            if k is not None:
                k += i
                generator_matrix[[i,k]] = generator_matrix[[k,i]]
                for m in range(nrows):
                    if m != i:
                        if generator_matrix[m][j] in (1,2):
                            generator_matrix[m] = self.rowmult(generator_matrix[i], generator_matrix[m])
                i += 1
        nondiagonal_rows = i
        for j in range(nrows):
            k = self.find_z_row_index(generator_matrix[i:], nrows-i, j)
            if k is not None:
                k += i
                generator_matrix[[i,k]] = generator_matrix[[k,i]]
                for m in range(nrows):
                    if m != i:
                        if generator_matrix[m,j] in (2,3):
                            generator_matrix[m] = self.rowmult(generator_matrix[i], generator_matrix[m])
                i += 1
        return generator_matrix, nondiagonal_rows

    def get_statevector(self):
        '''Obtain statevector corresponding to current stabilizer.
        IN:
            self: StabilizerState instance
        OUT:
            statevector: vector representation of state (exponentially large)
        '''
        matrix, circuit = self.basis_norm_circuit()
        matrix, circuit = self.circuit_to_all_zero_state(matrix, circuit)
        circ = QuantumCircuit(self.nrows)
        for c in circuit:
            if c[0] == 'H':
                circ.h(c[1])
            if c[0] == 'X':
                circ.x(c[1])
            if c[0] == 'CN':
                circ.cx(c[1], c[2])
            if c[0] == 'CP':
                circ.cz(c[1], c[2])
            if c[0] ==  'P':
                circ.p(c[1])
        statevector = Statevector(circ).data

        return statevector
    @staticmethod
    def conjugate_hadamard(matrix, nrows, j):
        '''Obtain generator matrix by conjugating by a Hadamard operation.
        IN:
            matrix: generator matrix to conjugate
            nrows: number of rows in generator matrix
            j: qubit on which to act with Hadamard
        OUT:
            matrix: generator matrix after conjugation.
        '''
        hadamard_transform = {0:0, 1:3, 2:6, 3:1}
        for i in range(nrows):
            matrix[i][j] = hadamard_transform[matrix[i][j]]
            #Transfer phase to phase column
            if matrix[i][j] > 3:
                matrix[i][j] -= 4
                matrix[i][-1] *= -1
        return matrix

    @staticmethod
    def conjugate_phase(matrix, nrows, j):
        '''Obtain generator matrix by conjugating by a Phase operation.
        IN:
            matrix: generator matrix to conjugate
            nrows: number of rows in generator matrix
            j: qubit on which to act with Phase
        OUT:
            matrix: generator matrix after conjugation.
        '''
        phase_transform = {0:0, 1:2, 2:5, 3:3}
        for i in range(nrows):
            matrix[i][j] = phase_transform[matrix[i][j]]
            #Transfer phase to phase column
            if matrix[i][j] > 3:
                matrix[i][j] -= 4
                matrix[i][-1] *= -1
        return matrix

    @staticmethod
    def conjugate_cnot(matrix, nrows, j, k):
        '''Obtain generator matrix by conjugating by a CNOT operation.
        IN:
            matrix: generator matrix to conjugate
            nrows: number of rows in generator matrix
            j: control qubit for CNOT
            k: target qubit for CNOT
        OUT:
            matrix: generator matrix after conjugation.
        '''
        for i in range(nrows):
            matrix[i][j], matrix[i][k] = cnot_transform[(matrix[i][j], matrix[i][k])]
            #Transfer phase to phase column
            if matrix[i][j] > 3:
                matrix[i][j] -= 4
                matrix[i][-1] *= -1
            if matrix[i][k] > 3:
                matrix[i][k] -= 4
                matrix[i][-1] *= -1
        return matrix

    @staticmethod
    def conjugate_cphase(matrix, nrows, j, k):
        '''Obtain generator matrix by conjugating by a CZ operation.
        IN:
            matrix: generator matrix to conjugate
            nrows: number of rows in generator matrix
            j: control qubit for CZ
            k: target qubit for CZ
        OUT:
            matrix: generator matrix after conjugation.
        '''
        for i in range(nrows):
            matrix[i][j], matrix[i][k] = cphase_transform[(matrix[i][j], matrix[i][k])]
            if matrix[i][j] > 3:
                matrix[i][-1] *= -1
                matrix[i][j] -= 4
            if matrix[i][k] > 3:
                matrix[i][-1] *= -1
                matrix[i][k] -= 4
        return matrix

    @staticmethod
    def conjugate_x(matrix, nrows, j):
        '''Obtain generator matrix by conjugating by a X operation.
        IN:
            matrix: generator matrix to conjugate
            nrows: number of rows in generator matrix
            j: qubit on which to act with X
        OUT:
            matrix: generator matrix after conjugation.
        '''
        x_transform = {0:0, 1:1, 2:6, 3:7}
        for i in range(nrows):
            matrix[i][j] = x_transform[matrix[i][j]]
            #Transfer phase to phase column
            if matrix[i][j] > 3:
                matrix[i][j] -= 4
                matrix[i][-1] *= -1
        return matrix

    @staticmethod
    def get_hadamard_phase(matrix, element, basis_state):
        '''Get the phase when applying a Hadamard to a basis state in a stabilizer.
        IN:
            matrix: generator matrix for stabiliser state
            element: qubit index on which to apply Hadamard
            basis_state: basis state included in stabiliser
        OUT:
            phase_h: phase when applying Hadamard to a basis state.
        '''
        basis_state2 = copy.deepcopy(basis_state)
        basis_state2.generator_matrix[element, -1] *= -1
        basis_state2.basis_state, basis_state2.basis_state_overlap = basis_state2.find_basis_state()
        state = StabilizerState(matrix)
        state.basis_state = copy.deepcopy(basis_state)
        local_phase, found = state.find_relative_phase_in_stab(state, basis_state2, state.nrows)
        if not(found) or math.isclose(np.imag(local_phase),0):
            return 0
        return local_phase

    def basis_norm_circuit_phase(self):
        '''Get circuit to transform generator matrix to computational basis form
        with phase
        IN:
            self: stabiliser state
        OUT:
            matrix: generator_matrix for resulting computational basis state.
            circuit: list of Clifford operations needed to transform to computational
                     basis
            phase: phase induced by basis transform.
        '''
        circuit = []
        matrix = self.generator_matrix.copy()
        nrows = self.nrows
        phase = 1+0j
        basis_state = copy.deepcopy(self.basis_state)
        for j in range(nrows):
            k = self.find_xy_row_index(matrix[j:], nrows-j,j)
            if k is not None:
                k+=j
                matrix[[k,j]] = matrix[[j,k]]
            else:
                k2 = self.find_last_z_row_index(matrix[j:], nrows-j, j)
                if k2 is not None:
                    k2 += j
                    matrix[[k2,j]] = matrix[[j,k2]]
                    if any([x in (1,2,3) for x in matrix[j][j+1:-1]]):
                        circuit.append(('H', j))
                        phase_h = self.get_hadamard_phase(matrix, j, basis_state)
                        matrix = self.conjugate_hadamard(matrix, nrows, j)
                        #Get resulting phase and basis state. Must be careful to
                        #keep the basis state that corresponds to the final state
                        #the circuit transforms to.
                        if matrix[j, -1] == -1 and list(matrix[j]).count(2)%2==0:
                            if basis_state.generator_matrix[j, -1] == -1:
                                phase *= (-1 + phase_h)/np.sqrt(1+abs(phase_h))
                            else:
                                phase *= (1 - phase_h)/np.sqrt(1+abs(phase_h))
                                basis_state.generator_matrix[j, -1] = -1
                        elif list(matrix[j]).count(2)%2==0:
                            if basis_state.generator_matrix[j, -1] == -1:
                                basis_state.generator_matrix[j, -1] = 1
                                phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
                            else:
                                phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
                        else:
                            if basis_state.generator_matrix[j, -1] == -1:
                                phase *= (-1 + phase_h)/np.sqrt(1+abs(phase_h))
                            else:
                                phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
        for j in range(nrows):
            for k in range(j+1, nrows):
                if matrix[j][k] in (1,2):
                    matrix = self.conjugate_cnot(matrix, nrows, j, k)
                    circuit.append(('CN', j, k))
                    if basis_state.generator_matrix[j, -1] == -1:
                        basis_state.generator_matrix[k, -1] *= -1

        for j in range(nrows):
            for k in range(j+1, nrows):
                if matrix[j][k] == 3:
                    matrix = self.conjugate_cphase(matrix, nrows, j, k)
                    circuit.append(('CP', j, k))
                    if basis_state.generator_matrix[j, -1] == -1 and basis_state.generator_matrix[k, -1] == -1:
                        phase *= -1

        for j in range(nrows):
            if matrix[j][j] == 2:
                matrix = self.conjugate_phase(matrix, nrows, j)
                if basis_state.generator_matrix[j, -1] == -1:
                    phase *= 1j
                circuit.append(('P', j))

        for j in range(nrows):
            if matrix[j][j] == 1:
                phase_h = self.get_hadamard_phase(matrix, j, basis_state)
                matrix = self.conjugate_hadamard(matrix, nrows, j)
                #Get resulting phase and basis state. Must be careful to
                #keep the basis state that corresponds to the final state
                #the circuit transforms to.
                if matrix[j, -1] == -1 and list(matrix[j]).count(2)%2==0:
                    if basis_state.generator_matrix[j, -1] == -1:
                        phase *= (-1 + phase_h)/np.sqrt(1+abs(phase_h))
                    else:
                        phase *= (1 - phase_h)/np.sqrt(1+abs(phase_h))
                        basis_state.generator_matrix[j, -1] = -1
                elif list(matrix[j]).count(2)%2==0:
                    if basis_state.generator_matrix[j, -1] == -1:
                        basis_state.generator_matrix[j, -1] = 1
                        phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
                    else:
                        phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
                else:
                    if basis_state.generator_matrix[j, -1] == -1:
                        phase *= (-1 + phase_h)/np.sqrt(1+abs(phase_h))
                    else:
                        phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
                    
                circuit.append(('H', j))
        return matrix, circuit, phase

    def basis_norm_circuit(self):
        '''Get circuit to transform generator matrix to computational basis form.
        IN:
            self: stabiliser state
        OUT:
            matrix: generator_matrix for resulting computational basis state.
            circuit: list of Clifford operations needed to transform to computational
                     basis
        '''
        circuit = []
        matrix = self.generator_matrix.copy()
        nrows = self.nrows
        for j in range(nrows):
            k = self.find_xy_row_index(matrix[j:], nrows-j,j)
            if k is not None:
                k+=j
                matrix[[k,j]] = matrix[[j,k]]
            else:
                k2 = self.find_last_z_row_index(matrix[j:], nrows-j, j)
                if k2 is not None:
                    k2 += j
                    matrix[[k2,j]] = matrix[[j,k2]]
                    if any([x in (1,2,3) for x in matrix[j][j+1:-1]]):
                        matrix = self.conjugate_hadamard(matrix, nrows, j)
                        circuit.append(('H', j))
        for j in range(nrows):
            for k in range(j+1, nrows):
                if matrix[j][k] in (1,2):
                    matrix = self.conjugate_cnot(matrix, nrows, j, k)
                    circuit.append(('CN', j, k))

        for j in range(nrows):
            for k in range(j+1, nrows):
                if matrix[j][k] == 3:
                    matrix = self.conjugate_cphase(matrix, nrows, j, k)
                    circuit.append(('CP', j, k))

        for j in range(nrows):
            if matrix[j][j] == 2:
                matrix = self.conjugate_phase(matrix, nrows, j)
                circuit.append(('P', j))

        for j in range(nrows):
            if matrix[j][j] == 1:
                matrix = self.conjugate_hadamard(matrix, nrows, j)
                circuit.append(('H', j))
        return matrix, circuit

    def circuit_to_all_zero_state(self, matrix, circuit):
        for j in range(self.nrows):
            if matrix[j][-1] == -1:
                circuit.append(('X', j))
                matrix[j][-1] = 1
        return matrix, circuit
                
        
    def conjugate_circuit(self, other, circuit):
        '''Conjugate the generator of another stabilizer state by a circuit.
        IN:
            self: stabilizer state
            other: stabilizer state
            circuit: circuit to conjugate by
        OUT:
            matrix: resulting generator matrix after conjugation
        '''
        #[TODO] this can be rewritten - static method? Or act on "self" not "other"
        matrix = other.generator_matrix.copy()
        nrows = len(matrix)
        for elem in circuit:
            match elem[0]:
                case 'H':
                    matrix = self.conjugate_hadamard(matrix, nrows, elem[1])
                case 'P':
                    matrix = self.conjugate_phase(matrix, nrows, elem[1])
                case 'X':
                    matrix = self.conjugate_x(matrix, nrows, elem[1])
                case 'CN':
                    matrix = self.conjugate_cnot(matrix, nrows, elem[1], elem[2])
                case 'CP':
                    matrix = self.conjugate_cphase(matrix, nrows, elem[1], elem[2])
        return matrix

    def conjugate_circuit_phase(self, other, circuit, final_basis_state):
        '''Conjugate the generator of another stabilizer state by a circuit
        with phase.
        IN:
            self: stabilizer state
            other: stabilizer state
            circuit: circuit to conjugate by
        OUT:
            matrix: resulting generator matrix after conjugation
            phase: phase induced by conjugation
        '''
        matrix = other.generator_matrix.copy()
        nrows = len(matrix)
        phase = 1+0j
        basis_state = StabilizerState(other.basis_state.generator_matrix.copy())
            
        for elem in circuit:
            match elem[0]:
                case 'H':
                    phase_h = self.get_hadamard_phase(matrix, elem[1], basis_state)
                    basis_state2 = copy.deepcopy(basis_state)
                    basis_state2.generator_matrix[elem[1], -1] *= -1
                    basis_state2.basis_state.generator_matrix[elem[1], -1] *= -1
                    ns = StabilizerState(matrix.copy())
                    lp1, fp1 = self.find_relative_phase_in_stab(ns, basis_state, nrows)
                    ns = StabilizerState(matrix.copy())
                    lp, fp = self.find_relative_phase_in_stab(ns, basis_state2, nrows)
                    matrix = self.conjugate_hadamard(matrix, nrows, elem[1])
                    lp /=lp1
                    #Get resulting phase and basis state. Must be careful to
                    #keep the basis state that corresponds to the final state
                    #the circuit transforms to.
                    if not phase_h:
                     if not fp:
                         if  basis_state.generator_matrix[elem[1], -1] == -1:
                             phase *= -1
                         else:
                             phase *= 1 
                     else:
                         if basis_state.generator_matrix[elem[1], -1] == 1:
                             if lp == -1:
                                 basis_state.generator_matrix[elem[1], -1] *= -1
                                 basis_state.basis_state.generator_matrix[elem[1], -1] *= -1
                         else:
                             if lp == 1:
                                 basis_state.generator_matrix[elem[1], -1] *= -1
                                 basis_state.basis_state.generator_matrix[elem[1], -1] *= -1
                             else:
                                 phase *= -1
                    else:
                        if basis_state.generator_matrix[elem[1], -1] == -1:
                            phase *= (-1 + phase_h)/np.sqrt(1+abs(phase_h))
                        else:
                            phase *= (1 + phase_h)/np.sqrt(1+abs(phase_h))
                case 'P':
                    matrix = self.conjugate_phase(matrix, nrows, elem[1])
                    if basis_state.generator_matrix[elem[1], -1] == -1:
                        phase *= 1j
                case 'CN':
                    matrix = self.conjugate_cnot(matrix, nrows, elem[1], elem[2])
                    if basis_state.generator_matrix[elem[1], -1] == -1:
                        basis_state.generator_matrix[elem[2], -1] *= -1
                        basis_state.basis_state.generator_matrix[elem[2], -1] *= -1
                case 'CP':
                    matrix = self.conjugate_cphase(matrix, nrows, elem[1], elem[2])
                    if basis_state.generator_matrix[elem[1], -1] == -1 and basis_state.generator_matrix[elem[2], -1] == -1:
                        phase *= -1
                case 'X':
                    matrix = self.conjugate_x(matrix, nrows, elem[1])
                    basis_state.generator_matrix[elem[1], -1] *= -1

        new_state = StabilizerState(matrix)
        basis_state.basis_state, basis_state.basis_state_overlap = basis_state.find_basis_state()
        found1 = True
        found2 = True
        if basis_state != new_state.basis_state:
            local_phase, found1 = self.find_relative_phase_in_stab(new_state, basis_state, nrows)
            phase /= local_phase
            basis_state = new_state.basis_state
        if basis_state !=final_basis_state:
            local_phase, found2 = self.find_relative_phase_in_stab(new_state, final_basis_state, nrows)
            phase *= local_phase
        return matrix, phase, (found1 and found2)

    @staticmethod
    def find_relative_phase_in_stab2(state1, state2, nrows):
        '''Find the relative phase of two basis states (state1.basis_state and state2.basis_state) in
        state1 and rewrite state1 to store state2.basis_state as its basis_state
        IN:
            state1, state2: statbiliser states
            nrows: number of rows in generator matrix
        OUT:
            found: if True state2.basis_state is included in state1
            local_phase: phase of state2.basis_state in state1
        '''
        gen_mat = state1.basis_state.generator_matrix.copy()
        row_effect = np.array([0]*(nrows)+[1], dtype=np.complex128)
        for row in range(nrows):
            x = -1
            for j in range(nrows):
                if state1.generator_matrix[row, j] in (1,2):
                    x = j
                    break
            if x > -1 and state1.basis_state.generator_matrix[x, -1] != state2.basis_state.generator_matrix[x, -1]:
                row_effect = state1.rowmult(state1.generator_matrix[row], row_effect)
                for i in range(nrows):
                    if state1.generator_matrix[row,i] in (1,2):
                        state1.basis_state.generator_matrix[i,-1] *= -1
        if state1.basis_state == state2.basis_state:
            state1.basis_state.generator_matrix = gen_mat.copy()
            local_phase = row_effect[-1]
            for i in range(nrows):
                if int(row_effect[i]) == 2:
                    if state1.basis_state.generator_matrix[i, -1] == 1:
                        local_phase *= 1j
                    else:
                        local_phase *= -1j
                if int(row_effect[i]) == 3:
                    if state1.basis_state.generator_matrix[i, -1] == -1:
                        local_phase *= -1
            return local_phase, True        
        state1.basis_state.generator_matrix = gen_mat.copy()
        return 1+0j, False

    @staticmethod
    def find_relative_phase_in_stab(state1, state2, nrows):
        '''Find the relative phase of two basis states (state1.basis_state and state2.basis_state) in
        state1 and rewrite state1 to store state2.basis_state as its basis_state
        IN:
            state1, state2: statbiliser states
            nrows: number of rows in generator matrix
        OUT:
            found: if True state2.basis_state is included in state1
            local_phase: phase of state2.basis_state in state1
        '''
        gen_mat = state1.basis_state.generator_matrix.copy()
        row_effect = np.array([0]*(nrows)+[1], dtype=np.complex128)
        for row in range(nrows):
            x = -1
            for j in range(nrows):
                if state1.generator_matrix[row, j] in (1,2):
                    x = j
                    break
            if x > -1 and state1.basis_state.generator_matrix[x, -1] != state2.basis_state.generator_matrix[x, -1]:
                row_effect = state1.rowmult(state1.generator_matrix[row], row_effect)
                for i in range(nrows):
                    if state1.generator_matrix[row,i] in (1,2):
                        state1.basis_state.generator_matrix[i,-1] *= -1
        if state1.basis_state == state2.basis_state:
            state1.basis_state.generator_matrix = gen_mat.copy()
            local_phase = row_effect[-1]
            for i in range(nrows):
                if int(row_effect[i]) == 2:
                    if state1.basis_state.generator_matrix[i, -1] == 1:
                        local_phase *= 1j
                    else:
                        local_phase *= -1j
                if int(row_effect[i]) == 3:
                    if state1.basis_state.generator_matrix[i, -1] == -1:
                        local_phase *= -1
            return local_phase, True        
        state1.basis_state.generator_matrix = gen_mat.copy()
        return 1+0j, False

    def inner_product(self, other, phase = False, return_k = False):
        '''Compute the (phased) inner product of two stabilizer states/
        IN:
            self, other: stabiliser states.
            phase: if True compute stabiliser phase.
        OUT:
            inner product of the two states.
        '''
        if not isinstance(other, StabilizerState):
            return NotImplemented
        if self.nrows != other.nrows:
            return NotImplemented

        nrows = self.nrows
        other_local = copy.deepcopy(other)
        self_local = copy.deepcopy(self)
        if phase:
            matrix, circuit= self_local.basis_norm_circuit()
            final_basis_state = StabilizerState(matrix.copy(), True)
            matrix, _ = self_local.canonicalize(matrix)
            matrix, phase1, valid1 = self_local.conjugate_circuit_phase(self_local, circuit, final_basis_state)
            final_basis_state = StabilizerState(matrix.copy(), True)
            matrix2, phase2, valid2 = self_local.conjugate_circuit_phase(other_local, circuit, final_basis_state)
        else:
            matrix, circuit = self_local.basis_norm_circuit()
            matrix2 = self_local.conjugate_circuit(other_local, circuit)
        matrix2, _ = self_local.canonicalize(matrix2)
        

        k = 0
        for row in range(nrows):
            if any([x in (1,2) for x in matrix2[row][:-1]]):
                k += 1
            else:
                r = np.zeros(nrows+1, dtype = np.int8)
                r[-1] = 1 
                for j in range(nrows):
                    if matrix2[row][j] == 3:
                        for l in range(nrows):
                            r[l] = pauli_multiplication_matrix[int(matrix[j][l]), int(r[l])]
                            if r[l] > 3:
                                r[l] -= 4
                                r[-1] *= -1
                        r[-1] *= matrix[j][-1]
                if all(r[:-1] == matrix2[row][:-1]) and r[-1] == - matrix2[row][-1]:
                    return 0
        if phase:
            if not (valid1 and valid2):
                return -2
            return np.conj(phase1) * phase2*2**(-k/2)
        if return_k:
            return k
        return 2**(-k/2)

    def find_basis_state(self):
        '''Find and store a computational basis state with non-zero overlap with the stabilizer state.
        IN:
            self: stabilizer state
        '''
        matrix = np.ndarray((self.nrows - self.m, self.nrows+1), dtype=bool)
        k = 0
        matrix[:,:] = False
        for i in range(self.m, self.nrows):
            for j in range(self.nrows):
                if self.generator_matrix[i][j] == 3:
                    matrix[k,j] = True
                if self.generator_matrix[i][j] in (1,2):
                    return ValueError
            if self.generator_matrix[i][-1] == -1:
                matrix[k][-1] = True
            else:
                matrix[k][-1] = False
            k += 1
        basis_state = ['Unknown']*self.nrows
        for row in range(self.nrows - self.m-1, -1,-1):
            for j in range(self.nrows-1, -1, -1):
                if matrix[row][j]:
                    if basis_state[j] == "Unknown":
                        basis_state[j] = matrix[row][-1]
                        matrix[row][-1] ^= matrix[row][-1]
                    else:
                        if any(matrix[row][:j]):
                            matrix[row][-1] ^= basis_state[j]
                        else:
                            return ValueError
        for i in range(self.nrows):
            if basis_state[i] == "Unknown":
                basis_state[i] = np.False_
        generator_matrix = np.array([[0]*i+[3]+[0]*(self.nrows-i-1)+[1] for i in range(self.nrows)], dtype =np.int8)
        for i in range(self.nrows):
            if basis_state[i]:
                generator_matrix[i][-1] *= -1
        basis_state = StabilizerState(generator_matrix, False)
        overlap = 1
        return basis_state, overlap
 
    def apply_hamiltonian(self, hamil):
        '''Apply a Hamiltonian to the stabilizer state.
        IN:
            self: stabilizer state.
            hamil: hamiltonian expressed as a linear combination of Paulis.
        OUT:
            local_states: resulting stabilizer states.
        '''
        local_states = {}
        for term in hamil:
            state = copy.deepcopy(self)
            basis_state = copy.deepcopy(self.basis_state)
            basis_state.basis_state = StabilizerState(basis_state.generator_matrix)
            phase = state.get_pauli_phase(state.basis_state.generator_matrix, term[1])
            state.conjugate_pauli(state.generator_matrix,term[1])
            state.basis_state, state.basis_state_overlap = state.find_basis_state()
            basis_state.conjugate_pauli(basis_state.generator_matrix, term[1])
            if basis_state != state.basis_state:
                local_phase, found = self.find_relative_phase_in_stab(state, basis_state, self.nrows)
                phase /= local_phase
            else:
                local_phase = 1
            if state in local_states:
                local_states[state] += term[0]*phase
            else:
                local_states[state] = term[0]*phase
        del_states = []
        for state in local_states:
            if math.isclose(abs(local_states[state]), 0):
                del_states.append(state)
        for state in del_states:
            del(local_states[state])
        return local_states

    @staticmethod
    def get_pauli_phase(matrix, term):
        '''Get phase of applying a Pauli string to a stabilizer state.
        IN:
            matrix: generator matrix for stabilizer state.
            term: Pauli string to apply.
        OUT:
            phase: resulting phase
        '''
        phase = 1
        for j in range(len(term)):
            if term[j] == 'Z':
                if matrix[j][-1] == -1:
                    phase *= -1
            elif term[j] == 'Y':
                if matrix[j][-1] == 1:
                    phase*= -1j
                else:
                    phase*=1j
        return phase 

    def conjugate_pauli(self, matrix, pauli):
        '''Conjugate a generator matrix by a Pauli string.
        IN:
            self: stabilizer state
            matrix: generator matrix to conjugate
            pauli: Pauli string by which to conjugate
        OUT:
            matrix: matrix after conjugation.
        ''' 
        paulis = {'X':1, 'Y':2, 'Z':3, 'I':0}
        nrows = self.nrows
        for i in range(nrows):
            phase = 1+0j
            for j in range(len(pauli)):
                matrix[i][j] = pauli_multiplication_matrix[matrix[i][j],paulis[pauli[j]]]
                #Transfer phase to phase column
                if 3<matrix[i][j] < 8:
                    matrix[i][j] -= 4
                    phase *= -1
                if 7<matrix[i][j] < 12:
                    matrix[i][j] -= 8
                    phase *= 1j
                if 11<matrix[i][j] < 16:
                    matrix[i][j] -= 12
                    phase *= -1j
                matrix[i][j] = pauli_multiplication_matrix[paulis[pauli[j]],matrix[i][j]]
                #Transfer phase to phase column
                if 3<matrix[i][j] < 8:
                    matrix[i][j] -= 4
                    phase *= -1
                if 7<matrix[i][j] < 12:
                    matrix[i][j] -= 8
                    phase *= 1j
                if 11<matrix[i][j] < 16:
                    matrix[i][j] -= 12
                    phase *= -1j
            matrix[i][-1] *= phase
        return matrix

    def conjugate_row(self, matrix, row):
        '''Conjugate a generator matrix by a Pauli string.
        IN:
            self: stabilizer state
            matrix: generator matrix to conjugate
            pauli: Pauli string by which to conjugate
        OUT:
            matrix: matrix after conjugation.
        ''' 
        nrows = self.nrows
        for i in range(nrows):
            phase = 1+0j
            for j in range(len(row)-1):
                matrix[i][j] = pauli_multiplication_matrix[matrix[i][j],int(row[j])]
                #Transfer phase to phase column
                if 3<matrix[i][j] < 8:
                    matrix[i][j] -= 4
                    phase *= -1
                if 7<matrix[i][j] < 12:
                    matrix[i][j] -= 8
                    phase *= 1j
                if 11<matrix[i][j] < 16:
                    matrix[i][j] -= 12
                    phase *= -1j
                matrix[i][j] = pauli_multiplication_matrix[int(row[j]),matrix[i][j]]
                #Transfer phase to phase column
                if 3<matrix[i][j] < 8:
                    matrix[i][j] -= 4
                    phase *= -1
                if 7<matrix[i][j] < 12:
                    matrix[i][j] -= 8
                    phase *= 1j
                if 11<matrix[i][j] < 16:
                    matrix[i][j] -= 12
                    phase *= -1j
            matrix[i][-1] *= phase
        return matrix

    def merge_states(self, state, relative):
        '''Merge two stabiliser states into a single state.
        IN:
            self: stabilizer state
            state: stabilizer state to merge with
            relative: relative phase of the two states
        OUT:
            new_state: merged stabilizer state
        '''
        matrix, circuit = self.basis_norm_circuit()
        matrix, circuit = self.circuit_to_all_zero_state(matrix, circuit)
        final_basis_state=StabilizerState(matrix, False)
        matrix2, phase = self.conjugate_circuit_phase(state, circuit, final_basis_state)
        total_phase = relative * phase
        if (matrix2[:,:-1] == 1).any() or (matrix2[:,:-1] == 2).any():
            return False, None
        new_matrix = np.zeros(matrix2.shape, dtype=np.int8)
        zero_ks = []
        one_ks = []
        for j in range(self.nrows):
            if matrix2[j][-1] == 1:
                zero_ks.append(j)
            else:
                one_ks.append(j) 
        i = 0
        for k in zero_ks:
            new_matrix[i][k] = 3
            new_matrix[i][-1] = 1
            i += 1
        if np.imag(total_phase) == 0:
            for k in one_ks:
                new_matrix[i][k] = 1
                new_matrix[i][-1] = int(np.real(total_phase))
            i+=1
            for j, k in enumerate(one_ks):
                if j < len(one_ks) - 1:
                    new_matrix[i][k] = 3
                    new_matrix[i][one_ks[j+1]] = 3
                    new_matrix[i][-1] = 1
                    i += 1

        elif np.real(total_phase) == 0:
            for k in one_ks:
                new_matrix[i][k] = 2
                new_matrix[i][-1] = total_phase/(0+1j)
                i+=1
            for j, k in enumerate(one_ks):
                if j < len(one_ks) - 1:
                    new_matrix[i][k] = 3
                    new_matrix[i][one_ks[j+1]] = 3
                    new_matrix[i][-1] = 1
                    i += 1
        new_state = StabilizerState(new_matrix, find_basis_state=True)
        circuit = circuit[::-1]
        new_matrix_final, phase_final = self.conjugate_circuit_phase(new_state, circuit, new_state.basis_state)
        new_state = StabilizerState(new_matrix_final, find_basis_state = True)
        new_state.basis_state_overlap *= phase_final
        return True, new_state
        
    def compute_eproj(self, states, hamiltonian):
        '''Compute projected energy of a linear combination of stabilizers onto the current state.
        IN:
            self: stabilizer state
            states: linear combination of stabilizers
            hamiltonian: Hamiltonian operator expressed as a linear combination of Paulis.
        OUT:
            e_proj: projected energy
        '''
        e_proj = 0
        overlap = 0
        for state in states: 
            new_states = state.apply_hamiltonian(hamiltonian)
            for state2 in new_states:
                e_proj += self.inner_product(state2, True)*new_states[state2]*states[state]
            s = self.inner_product(state, True)
            overlap += s*states[state]
        return e_proj/overlap
            
# Nearest neighbour decomposition functions. Not well-tested.
# -----------------------------------------------------------
    def decompose_nearest_neighbours(self, j):
        row_j = -1
        matrix = self.generator_matrix
        matrix1 = self.generator_matrix.copy()
        matrix2 = self.generator_matrix.copy() 
        Zj = np.array([0]*j+[3]+[0]*(self.nrows-j-1)+[1])
        
        for row in range(self.nrows):
            if matrix[row][j] == 1 or matrix[row][j] == 2:
                if row_j == -1:
                    row_j = row
                else:
                    matrix1[row] = self.rowmult(matrix[row_j], matrix[j]) 
                    matrix2[row] = self.rowmult(matrix[row_j], matrix[j]) 
        matrix1[row_j] = Zj
        matrix2[row_j] = Zj
        matrix2[row_j][-1] = -1

        state1 = StabilizerState(matrix1, True)
        state2 = StabilizerState(matrix2, True)
        
        nrows = self.nrows
        local_phase=1
        if state1.basis_state != self.basis_state:
           local_phase, found = self.find_relative_phase_in_stab(state1, self, nrows)
           if found:
               state1.basis_state_overlap *= local_phase
           else:
               local_phase, found = self.find_relative_phase_in_stab(self, state1, nrows)
               self.basis_state_overlap *= local_phase
        matrix, circuit, phase11 = self.basis_norm_circuit_phase()
        final_basis_state = StabilizerState(matrix, False)
        matrix2, phase21 = self.conjugate_circuit_phase(state1, circuit, final_basis_state)
        phase1=np.conj(phase11*self.basis_state_overlap) * phase21*state1.basis_state_overlap
        if state2.basis_state != self.basis_state:
           local_phase, found = self.find_relative_phase_in_stab(state2, self, nrows)
           if found:
               state2.basis_state_overlap *= local_phase
           else:
               local_phase, found = self.find_relative_phase_in_stab(self, state2, nrows)
               self.basis_state_overlap *= local_phase
        matrix, circuit, phase12 = self.basis_norm_circuit_phase()
        final_basis_state = StabilizerState(matrix, False)
        matrix2, phase22 = self.conjugate_circuit_phase(state2, circuit, final_basis_state)
        phase2=np.conj(phase12*self.basis_state_overlap) * phase22*state2.basis_state_overlap
        return state1, state2, phase1, phase2 
                
    def decompose_nearest_neighbours2(self, j):
        row_j = -1
        matrix = self.generator_matrix
        matrix1 = self.generator_matrix.copy()
        matrix2 = self.generator_matrix.copy() 
        Xj = np.array([0]*j+[1]+[0]*(self.nrows-j-1)+[1])
        
        for row in range(self.nrows):
            if matrix[row][j] == 1 or matrix[row][j] == 2:
                return False, False, False, False
            if matrix[row][j] == 3:
                if row_j == -1:
                    row_j = row
                else:
                    matrix1[row] = self.rowmult(matrix[row_j], matrix[j]) 
                    matrix2[row] = self.rowmult(matrix[row_j], matrix[j]) 
        matrix1[row_j] = Xj
        matrix2[row_j] = Xj
        matrix2[row_j][-1] = -1

        state1 = StabilizerState(matrix1, True)
        state2 = StabilizerState(matrix2, True)
        
        nrows = self.nrows
        local_phase=1
        if state1.basis_state != self.basis_state:
           local_phase, found = self.find_relative_phase_in_stab(state1, self, nrows)
           if found:
               state1.basis_state_overlap *= local_phase
           else:
               local_phase, found = self.find_relative_phase_in_stab(self, state1, nrows)
               self.basis_state_overlap *= local_phase
        matrix, circuit, phase11 = self.basis_norm_circuit_phase()
        final_basis_state = StabilizerState(matrix, False)
        matrix2, phase21 = self.conjugate_circuit_phase(state1, circuit, final_basis_state)
        phase1=np.conj(phase11*self.basis_state_overlap) * phase21*state1.basis_state_overlap
        if state2.basis_state != self.basis_state:
           local_phase, found = self.find_relative_phase_in_stab(state2, self, nrows)
           if found:
               state2.basis_state_overlap *= local_phase
           else:
               local_phase, found = self.find_relative_phase_in_stab(self, state2, nrows)
               self.basis_state_overlap *= local_phase
        matrix, circuit, phase12 = self.basis_norm_circuit_phase()
        final_basis_state = StabilizerState(matrix, False)
        matrix2, phase22 = self.conjugate_circuit_phase(state2, circuit, final_basis_state)
        phase2=np.conj(phase12*self.basis_state_overlap) * phase22*state2.basis_state_overlap
        return state1, state2, phase1, phase2 
# -----------------------------------------------------------

# Stabilizer orthogonalisation functions. Not well-tested.
# -----------------------------------------------------------
def pauli_column(state, j):
    matrix = state.generator_matrix
    a = matrix[:,j]
    rows = [i for i in range(len(a)) if a[i] != 0]
    if all([x in (0,3) for x in a]):
        return 0, rows
    if all([x in (0,1) for x in a]):
        return 1, rows
    if all([x in (0,2) for x in a]):
        return 2, rows
    if all([x in (0,1,3) for x in a]):
        return 3, rows
    if all([x in (0,2,3) for x in a]):
        return 4, rows
    else:
        return 5, rows
def orthogonalise(states, coeffs):
    nrows = states[0].nrows
    for j in range(nrows):
        b = 0
        length_of_states = len(states)
        l, rows1 = pauli_column(states[0], j)
        for i in range(1, length_of_states):
            k, rows2 = pauli_column(states[i], j)
            if k != l or not((rows1==rows2)):
                b = 1
                break
        if b == 1:
            index = 0
            for i in range(length_of_states):
                k, rows = pauli_column(states[index], j)
                if k != 0:
                    state1, state2, alpha, beta = states[index].decompose_nearest_neighbours(j)
                    f1 = False
                    f2 = False
                    for q in range(len(states)):
                        if not(f1):
                            if states[q] == state1:
                                f1 = True
                                coeffs[q] += coeffs[index]*alpha/np.sqrt(2)
                        if not(f2):
                            if states[q] == state2:
                                f2 = True
                                coeffs[q] += coeffs[index]*beta/np.sqrt(2)
                        if f1 and f2:
                            break
                    if not f1:
                        states.append(state1)
                        coeffs.append(coeffs[index]*alpha/np.sqrt(2))
                    if not f2:
                        states.append(state2)
                        coeffs.append(coeffs[index]*beta/np.sqrt(2))
                    del(states[index])
                    del(coeffs[index])
                else:
                    index += 1
def orthogonalise2(states, coeffs):
    nrows = states[0].nrows
    maxm = max(x.m for x in states)
    for j in range(nrows):
        b = 0
        length_of_states = len(states)
        l, rows1 = pauli_column(states[0], j)
        for i in range(1, length_of_states):
            k, rows2 = pauli_column(states[i], j)
            if k != l or not((rows1==rows2)):
                b = 1
                break
        if b == 1:
            index = 0
            for i in range(length_of_states):
                k, rows = pauli_column(states[index], j)
                if k == 0:
                     state1, state2, alpha, beta = states[index].decompose_nearest_neighbours2(j)
                     if state1 != False:
                        f1 = False
                        f2 = False
                        for q in range(len(states)):
                            if not(f1):
                                if states[q] == state1:
                                    f1 = True
                                    coeffs[q] += coeffs[index]*alpha/np.sqrt(2)
                            if not(f2):
                                if states[q] == state2:
                                    f2 = True
                                    coeffs[q] += coeffs[index]*beta/np.sqrt(2)
                            if f1 and f2:
                                break
                        if not f1:
                            states.append(state1)
                            coeffs.append(coeffs[index]*alpha/np.sqrt(2))
                        if not f2:
                            states.append(state2)
                            coeffs.append(coeffs[index]*beta/np.sqrt(2))
                        del(states[index])
                        del(coeffs[index])
                else:
                     index += 1
# -----------------------------------------------------------

def compute_e_var(states, hamiltonian):
    '''Compute variational energy of a linear combination of stabilizers.
    IN:
        states: linear combination of stabilizers
        hamiltonian: Hamiltonian operator expressed as a linear combination of Paulis.
    OUT:
        e_var: variational energy
    '''
    e = 0
    o = 0
    e_contrib = []
    
    for state1 in states:
        new_states = state1.apply_hamiltonian(hamiltonian)
        for state2 in states:
            for state3 in new_states:
                e += state2.inner_product(state3, True)*new_states[state3]*states[state1]*states[state2]
                e_contrib.append(state2.inner_product(state3, True)*new_states[state3]*states[state1]*states[state2])
            o += state2.inner_product(state1, True)*states[state1]*states[state2]
    e_var = e/o
    return e_var



if __name__ == "__main__":

    from shades.shadows import CliffordGroup

    ensemble = CliffordGroup(4)
    cliff = ensemble.generate_sample()

    StabilizerState.from_stim_tableau(cliff)