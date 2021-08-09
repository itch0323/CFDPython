from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import random

class calc():
    def __init__(self):
        self.nx, self.ny = 128, 128
        self.spx, self.epx, self.spy, self.epy = self.obstacle(nx), self.obstacle(ny)
        self.dx, self.dy = 2/(self.nx - 1), 2/(self.ny - 1)
        self.nt = 100
        self.nit = 100
        self.rho = 1
        self.nu = .1
        self.dt = .0001

    def obstacle(self, mesh):
        spx = random.randint(mesh/8, 3*mesh/4)
        return int(spx), int(spx+(mesh/8))

    def build_up_b(self, b, u, v):
        
        b[1:-1, 1:-1] = (self.rho * (1 / self.dt * 
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                        (2 * self.dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * self.dy)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * self.dx))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * self.dy) *
                            (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * self.dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * self.dy))**2))
        return b

    def pressure_poisson(self, p, b):
        pn = np.empty_like(p)
        pn = p.copy()
        
        for q in range(self.nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx**2) /
                            (2 * (self.dx**2 + self.dy**2)) -
                            self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * 
                            b[1:-1,1:-1])

            p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
            p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
            
            p[self.spx, self.spy:self.epy] = p[self.spx-1, self.spy:self.epy]
            p[self.epx, self.spy:self.epy] = p[self.epx-1, self.spy:self.epy]
            p[self.spx:self.epx, self.spy] = p[self.spx:self.epx, self.spy-1]
            p[self.spx:self.epx, self.epy] = p[self.spx:self.epx, self.epy-1]
            p[self.spx+1:self.epx-1, self.spy+1:self.epy-1] = 0
        return p

    def cavity_flow(self):
        u, v = np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx))
        p, b = np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx))
        un , vn = np.empty_like(u), np.empty_like(v)
        
        for n in range(self.nt):
            un, vn = u.copy(), v.copy()
            b = self.build_up_b(b, u, v)
            p = self.pressure_poisson(p, b)
            
            u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                            un[1:-1, 1:-1] * self.dt / self.dx *
                            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                            vn[1:-1, 1:-1] * self.dt / self.dy *
                            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                            self.dt / (2 * self.rho * self.dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                            self.nu * (self.dt / self.dx**2 *
                            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                            self.dt / self.dy**2 *
                            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                            un[1:-1, 1:-1] * self.dt / self.dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                            vn[1:-1, 1:-1] * self.dt / self.dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                            self.dt / (2 * self.rho * self.dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                            self.nu * (self.dt / self.dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                            self.dt / self.dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :], v[0, :] = 0, 0
            u[:, 0], v[:, 0] = 0.98, 0.02
            u[:, -1], v[:, -1] = 0, 0
            u[-1, :], v[-1, :] = 0, 0
            u[self.spx:self.epx, self.spy:self.epy], v[self.spx:self.epx, self.spy:self.epy] = 0, 0

        X, Y = np.meshgrid(np.linspace(0, 2, self.nx), np.linspace(0, 2, self.ny))
        fig = plt.figure(figsize=(11,7), dpi=100)

        plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
        plt.colorbar()
        plt.contour(X, Y, p, cmap=cm.viridis)
        plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

def main():
    c = calc()
    c.cavity_flow()
 
if __name__ == '__main__':
    main()