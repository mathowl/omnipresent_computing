import numpy as np
import matplotlib.pyplot as plt
import shelve   
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

def LorenzV(t,X,s=10, r=28, b=2.667):
    x,y,z = X[0],X[1],X[2]
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


def rk4(f,t,x,h):
    k1 = h * f(t,x)
    k2 = h * f(t,x + 0.5 * k1)
    k3 = h * f(t,x + 0.5 * k2)
    k4 = h * f(t,x + k3)
    return x + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)


def mackey_glass(length=10000, x0=None, a=0.2, b=0.1, c=10.0, tau=23.0,
                 n=1000, sample=1, discard=250):
    
    """edited code based on https://github.com/manu-mannattil/nolitsa
    
    (see Appendix A in https://www.sciencedirect.com/science/article/pii/0167278983902981)

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : int, (default = 1)
        Sampling step of the time series. Pick somewhere between n/10 and n/100.
    discard : int, optional (default = 250)
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    np.random.seed(0)
    grids = discard + sample * length +1
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n)) 
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[discard::sample]

def mg_datasets(tau=17,length=1_000_000,n=1000,sample=1,discard=4000):
    y=mackey_glass(length=length,tau=tau,n=n,sample=sample,discard=discard)
    return y

def RoesV(t,X,a = 0.2, b = 0.2, c = 5.7):
    x,y,z = X[0],X[1],X[2]
    return np.array([- y - z, x + a * y, b + z * (x - c)])



def roes_datasets():
    dt=1/2000
    T=np.arange(0,1501+dt,dt)
    N=len(T)
    X=np.zeros((3,N))
    x=np.array( [1,1,1])/10
    X[:,0] = x 
    for i in range(1,N):
        x = rk4(RoesV,T[i],x,dt)
        X[:,i] = x 
    return X[:,50000:]



def loz_datasets():
    dt=1/2000
    T=np.arange(0,1501+dt,dt)
    N=len(T)
    X=np.zeros((3,N))
    x=np.array( [1,1,1])/10
    X[:,0] = x 
    for i in range(1,N):
        x = rk4(LorenzV,T[i],x,dt)
        X[:,i] = x 
    return X[:,50000:]

def narma10(v,a=0.3,b=0.05,c=1.5,d=0.1,n=10):
    T = len(v)
    y = np.zeros(T)
    for t in range(10,T):
        y[t] = a * y[t-1] + b * y[t-1] * (np.sum(y[t-n:t])) + c * v[t-n] * v[t-1] + d
    return y



class KS_ETDRK2():
    def __init__(
        self,
        L,
        N,
        dt,
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N

        wavenumbers = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi))
        self.derivative_operator = 1j * wavenumbers

        linear_operator = - self.derivative_operator**2 - self.derivative_operator**4
        self.exp_term = jnp.exp(dt * linear_operator)
        self.coef_1 = jnp.where(
            linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_operator,
        )
        self.coef_2 = jnp.where(
            linear_operator == 0.0,
            dt / 2,
            (self.exp_term - 1.0 - linear_operator * dt) / (linear_operator**2 * dt)
        )

        self.alias_mask = (wavenumbers < 2/3 * jnp.max(wavenumbers))
    
    def __call__(
        self,
        u,
    ):
        u_nonlin = - 0.5 * u**2
        u_hat = jnp.fft.rfft(u)
        u_nonlin_hat = jnp.fft.rfft(u_nonlin)
        u_nonlin_hat = self.alias_mask * u_nonlin_hat
        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        u_stage_1_hat = self.exp_term * u_hat + self.coef_1 * u_nonlin_der_hat
        u_stage_1 = jnp.fft.irfft(u_stage_1_hat, n=self.N)

        u_stage_1_nonlin = - 0.5 * u_stage_1**2
        u_stage_1_nonlin_hat = jnp.fft.rfft(u_stage_1_nonlin)
        u_stage_1_nonlin_hat = self.alias_mask * u_stage_1_nonlin_hat
        u_stage_1_nonlin_der_hat = self.derivative_operator * u_stage_1_nonlin_hat

        u_next_hat = u_stage_1_hat + self.coef_2 * (u_stage_1_nonlin_der_hat - u_nonlin_der_hat)
        u_next = jnp.fft.irfft(u_next_hat, n=self.N)

        return u_next


def KS_datasets():
    """https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fft_and_spectral_methods/ks_solver_etd_and_etdrk2_in_jax.ipynb"""
    """I noticed some precision issues when I used Jax.jit so jit is not used 
    However, as I am using windows I am running a somewhat outdated version of Jax.jit so 
    perhaps it might work better on other os."""
    
    DOMAIN_SIZE =45.0
    N_DOF = 400
    DT = 0.05


    mesh = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)
    per = 4
    u_0 = jnp.sin(per * jnp.pi * mesh / DOMAIN_SIZE) 
    ks_stepper_etdrk2 = KS_ETDRK2(
        L=DOMAIN_SIZE,
        N=N_DOF,
        dt=DT,
    )
    u_current = u_0
    trj_etdrk2 = [u_current, ]
    for i in range(120_000):
        u_current = ks_stepper_etdrk2(u_current)
        trj_etdrk2.append(u_current)
    
    trj_etdrk2 = jnp.stack(trj_etdrk2)
    etdrk2_np = np.array(trj_etdrk2) 
    return etdrk2_np[8_000:,:]   


def narma10_datasets():
    np.random.seed(0)
    n=10
    u_asym = np.random.rand(20_000) 

    # NARMA10
    v = 0.2*u_asym 
    y = (narma10(v,n=n))

    y=y[15:]
    ut=u_asym[15:]   
    return y,ut


def sheet_music():
    degrees  = ['C','D','F','D','A','A','G',
                'C','D','F','D','G','G','F',\
                'C','D','F','D','F','G','E',\
                'D','C','C','C','G','F']
    beat = [1,1,1,1,3,3,6,\
            1,1,1,1,3,3,6,\
            1,1,1,1,4,2,3,\
            1,2,2,2,4,8]  

    let2int = {'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'A' : 6}
    mel_array= [let2int[lettre] for lettre in degrees]

    new_mel = []
    for _ in range(500):
        new_mel.append(mel_array)
    new_mel = np.array(new_mel).flatten()

    cubic_melo = CubicSpline(np.arange(0,len(new_mel),1),new_mel)
    cx = np.arange(0,len(new_mel)-0.1,0.1)

    new_beat = []
    for _ in range(500):
        new_beat.append(beat)
    new_beat = np.array(new_beat).flatten()

    cubic_beat = CubicSpline(np.arange(0,len(new_beat),1),new_beat)
    cx = np.arange(0,len(new_beat)-0.1,0.1)
    return cubic_melo(cx), cubic_beat(cx)




def main():
    yR= roes_datasets()
    yM = mg_datasets()
    yL = loz_datasets()
    yKS = KS_datasets()
    ynarma, unarma  = narma10_datasets()

    narma_dict = dict()
    narma_dict['input'] = unarma
    narma_dict['output'] = ynarma

    cm,cb = sheet_music()

    #checks:
    plt.plot(yM[::20][:2000])
    plt.savefig(f'data_pictures/MG.jpg',dpi=100,bbox_inches = 'tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot(yL[0,::20][:3000],yL[1,::20][:3000],yL[2,::20][:3000])
    plt.savefig(f'data_pictures/Lor.jpg',dpi=100,bbox_inches = 'tight')
    plt.close()

    plt.plot(ynarma[:500])
    plt.savefig(f'data_pictures/narma.jpg',dpi=100,bbox_inches = 'tight')
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot(yR[0,::50][:3000],yR[1,::50][:3000],yR[2,::50][:3000])
    plt.savefig(f'data_pictures/Roes.jpg',dpi=100,bbox_inches = 'tight')
    plt.close()



    plt.figure(figsize=(20, 5))
    plt.imshow(
        yKS.T,
        cmap="RdBu",
        aspect="auto",
        origin="lower",
        extent=(0, yKS.shape[0], 0, 100),
        vmin=-2,
        vmax=2,
    )
    plt.colorbar()
    plt.xlabel("steps")
    plt.ylabel("space")
    plt.savefig(f'data_pictures/KS.jpg',dpi=100,bbox_inches = 'tight')
    plt.close()

    #shelving

    for i in [2,4,5]:
        with shelve.open(f'supp/section{i}/data/data_sets.shelve', 'c') as shelf:
            shelf['lorenz'] =  yL

    with shelve.open(f'supp/section2/data/data_sets.shelve', 'c') as shelf:
        shelf['mackey_glass'] = yM 
        shelf['ks_equations'] = yKS
        shelf['narma']  = narma_dict


    with shelve.open(f'main/data/data_sets.shelve', 'c') as shelf:
        shelf['lorenz'] =  yL

    with shelve.open(f'supp/roesler/data/data_sets.shelve', 'c') as shelf:
        shelf['roes'] = yR 

    with shelve.open('supp/music_box/data/rick_roll.shelve', 'c') as shelf:
        shelf['degree'] =  cm
        shelf['beat'] =  cb


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()