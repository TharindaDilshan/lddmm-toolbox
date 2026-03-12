from torch.autograd import grad

# ──────────────────────────────────────────────
# ODE Integrator
# ──────────────────────────────────────────────
def RalstonIntegrator():
    """Second-order Ralston ODE integrator for tuple-valued systems."""
    def f(ODESystem, x0, nt=10, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)

        return l

    return f

# ──────────────────────────────────────────────
# Hamiltonian mechanics
# ──────────────────────────────────────────────
def Hamiltonian(K):
    def H(p, q):
        return 0.5 * (p * K(q, q, p)).sum()

    return H

def HamiltonianSystem(K):
    H = Hamiltonian(K)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)

        return -Gq, Gp

    return HS

# ──────────────────────────────────────────────
# Geodesic shooting and flow
# ──────────────────────────────────────────────
def Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator(), deltat=1.0):
    """Geodesic shooting from initial momentum p0 and position q0."""
    return Integrator(HamiltonianSystem(K), (p0, q0), nt, deltat)


def Flow(x0, p0, q0, K, nt=10, deltat=1.0, Integrator=RalstonIntegrator()):
    """Transport arbitrary points x0 along the geodesic defined by (p0, q0)."""
    HS = HamiltonianSystem(K)

    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), nt=nt, deltat=deltat)