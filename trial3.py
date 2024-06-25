import kwant
import numpy as np
import tinyarray
from matplotlib import pyplot as plt
import kwant.continuum
import scipy.sparse.linalg
import scipy.linalg 

# Constants:
h = 1
m1 = 0.00139
m2 = 1
d1 = 0
d2 = 0.34 * 1.602e-19
Ef1 = 0
Ef2 = 1.6e-18
V = 0.16
B = 0.26
phi = 0
g1 = -40 * 9.2e-24
g2 = 2*9.2e-24
ay = 0
az = 0

# Pauli matrices using tinyarray
zero_matrix = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])

# Hamiltonians with position-dependent potential (x)
hamiltonian1 = f"""
    ({h}**2)/(2 * {m1}) * (k_y**2 + k_x**2) * sigma_z - ({Ef1} + 1.6e-19 * phi * x) * sigma_z 
    + ay * x * (k_x * sigma_x) * sigma_z - az * x * k_y * sigma_x * sigma_z
    + 0.5 * {g1} * {B} * sigma_x + {d1} * sigma_x
"""
template1 = kwant.continuum.discretize(hamiltonian1)

hamiltonian2 = f"""
    ({h}**2)/(2 * {m2}) * (k_y**2 + k_x**2) * sigma_z - ({Ef2} + 1.6e-19 * phi * x) * sigma_z 
    + ay * x * (k_x * sigma_x) * sigma_z - az * x * k_y * sigma_x * sigma_z
    + 0.5 * {g2} * {B} * sigma_x + {d2} * sigma_x
"""
template2 = kwant.continuum.discretize(hamiltonian2)

def make_system(a=1, t=1.0, W=150, L=150):
    lat = kwant.lattice.square(a=1, name='', norbs=None)

    def hexagon(site):
        (x, y) = site.pos
        return (np.abs(x) < 15 and np.abs(y) < 15)

    sys = kwant.Builder()
    sys.fill(template1, hexagon, (0, 0))

    sym_lead = kwant.TranslationalSymmetry((a, 0))
    lead = kwant.Builder(sym_lead)

    def lead_shape(site):
        (x, y) = site.pos
        return np.abs(y) < 15

    lead.fill(template1, lead_shape, (0, 0))
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())

    lead2 = kwant.Builder(sym_lead)
    lead2.fill(template2, lead_shape, (0, 0))
    sys.attach_lead(lead2)

    kwant.plot(sys)

    return sys

def plot_differential_conductance(sys, gate_voltages):
    data = []
    for Vg in gate_voltages:
        params = dict(phi=1e10 * Vg, az = 0.75 * 1.602e-19 * Vg, ay = 1.602e-19 * Vg) 
        smatrix = kwant.smatrix(sys, energy=1.0, params=params)
        data.append(smatrix.transmission(1, 0)) 

    plt.figure()
    plt.plot(gate_voltages, data, marker='o')
    plt.xlabel("Gate Voltage")
    plt.ylabel("Differential Conductance")
    plt.title("Differential Conductance vs Gate Voltage (B = 0.36T)")
    plt.grid(True)
    plt.show()

def main():
    sys = make_system()
    sys = sys.finalized()
    gate_voltages = np.linspace(-10, 10, 100) 
    plot_differential_conductance(sys, gate_voltages)

if __name__ == '__main__':
    main()
