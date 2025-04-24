import numpy as np
import math
import shelve
import numba as nb
import time


@nb.njit()
def frk4_step(y,u0,umid,u1,h,w,K,F,N):
    k1 = h * Fkuramoto_nb( y, u0,w,K,F,N)
    k2 = h * Fkuramoto_nb(y + 0.5 * k1, umid,w,K,F,N)
    k3 = h * Fkuramoto_nb( y + 0.5 * k2, umid,w,K,F,N)
    k4 = h * Fkuramoto_nb(y + k3, u1, w,K,F,N)
    y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    return ((y) % (2*math.pi))

@nb.njit()
def Fkuramoto_nb(x,ut,w,K,F,N):
    y = np.empty((x.shape[0]),dtype='float64')
    R1=0
    R2=0
    for i in nb.prange(x.shape[0]):
        R2 +=np.cos(x[i])
        R1 +=np.sin(x[i])   
    for i in nb.prange(x.shape[0]):
        y[i] = w[i] - K*R2*np.sin(x[i])/N + K*R1*np.cos(x[i])/N +  F* np.sin(ut[i]-x[i])
    return y


@nb.njit()
def whipeout(N, initlen, ut0,utmid,ut1, x0, h,w,p):
    uin, K,F= p[0],p[1], p[2]
    
    
    for j in nb.prange(initlen):
        U0 = ut0[:,j]
        Umid = utmid[:,j]
        U1 = ut1[:,j]
        
        x  = frk4_step(x0,uin*U0,uin*Umid,uin*U1,h,w,K,F,N)
        x0 = x
    return x

@nb.njit()
def kura_run(N,runlen, ut0, utmid,ut1, x0, h,w,p):
    uin, K,F = p[0],p[1], p[2]
    X = np.empty((N,runlen))
    for j in nb.prange(runlen):
        U0 = ut0[:,j]
        Umid = utmid[:,j]
        U1 = ut1[:,j]
        
        x  = frk4_step(x0,uin*U0,uin*Umid,uin*U1,h,w,K,F,N)
        x0 = x
        X[:,j] = x
    return X

@nb.njit()
def nb_dot( x, y): 
    res = np.zeros(1,dtype = 'float64')
    for i in nb.prange(x.shape[0]):
        res+=x[i]*y[i]
    return res

@nb.njit()
def nb_concatdot3( x, y): 
    res = np.zeros(3,dtype = 'float64')
    for i in nb.prange(x.shape[1]):
        if i==0:
            res[0]+=x[0,i]
            res[1]+=x[1,i]
            res[2]+=x[2,i]
        else:
            res[0]+=x[0,i]*y[i-1,0]
            res[1]+=x[1,i]*y[i-1,0]
            res[2]+=x[2,i]*y[i-1,0]
    return res

@nb.njit()
def nb_concatdot30( x, y): 
    res = np.zeros(3,dtype = 'float64')
    for i in nb.prange(x.shape[1]):
        if i==0:
            res[0]+=x[0,i]
            res[1]+=x[1,i]
            res[2]+=x[2,i]
        else:
            res[0]+=x[0,i]*y[i-1]
            res[1]+=x[1,i]*y[i-1]
            res[2]+=x[2,i]*y[i-1]
    return res

@nb.njit()
def NMSE(yeval,ydata):
    
    nmse = np.zeros(yeval.shape[0], dtype='float64')
    for i in nb.prange(yeval.shape[0]):
        for j in nb.prange(yeval.shape[1]):
            nmse[i] += (ydata[i,j]-yeval[i,j])**2
        nmse[i]=nmse[i]/numba_norm(ydata[i,:])**2
    return nmse

@nb.njit()
def gram_schmidt(U,k):
    N = np.shape(U)[0]
    W = U.copy().astype(np.float64)
    V = np.empty((N, k),dtype='float64')
    norms = np.empty(k,dtype='float64')
    for i in range(k):
        for j in range(i):
            W[:, i] = W[:, i] - numba_dot2(U[:, i], V[:, j]) * V[:, j]
        norms[i] = numba_norm(W[:, i])
        V[:, i] = W[:, i] / norms[i]
    return V, norms 

@nb.njit() 
def numba_norm(a):
    n = a.shape[0]
    norm = 0
    for i in range(n):
        norm += a[i] * a[i]
    return np.sqrt(norm)


@nb.njit() 
def assemble(v, U, N,k):
    new_state = np.empty(N+N*k, dtype='float64')
    new_state[:N] = v 
    new_state[N:] = U.flatten()
    return new_state

@nb.njit() 
def disassemble(state, N,k):
    new_v = state[:N] % (2*math.pi)
    new_U = state[N:].reshape(N, k)
    return new_v, new_U


@nb.njit()
def dSdt( f,fjac, p,pvec,Up,W1,W2,Wout,w, t, state, N,k):
    v, U = disassemble(state, N,k)
    dv = f(t, v,p,pvec,Up,Wout ,w, N)
    dU = fjac(t, v,p, pvec,Up,W1,W2,Wout,w, N) @ U
    return assemble(dv, dU, N,k)


@nb.njit()
def varRK4(t, state, dt, f, fjac,p, pvec,Up,W1,W2,Wout,w,N,k):
    tmid = t + dt*0.5
    k1 = dt*dSdt( f, fjac, p, pvec,Up,W1,W2,Wout,w, t, state,  N,k)
    k2 = dt*dSdt(f,fjac,p, pvec,Up, W1,W2,Wout,w, tmid, state + 0.5 * k1, N,k)
    k3 = dt*dSdt(f, fjac,p,pvec,Up,W1,W2,Wout, w, tmid, state + 0.5 * k2,  N,k)
    k4 = dt*dSdt(f, fjac,p, pvec,Up,W1,W2,Wout,w, t + dt, state +  k3,  N,k)
    return state + (1/6) * (k1 + 2*k2 + 2*k3 + k4)


@nb.njit(parallel=True)
def Jkura_closed(t,x,p,pvec,Up,W1,W2,Wout,w,N):

    uin, K,F= p[0],p[1],p[2]
    one_hot = pvec
    
    J=np.zeros((N,N),dtype='float64')      
    Jupred=np.zeros((3,N),dtype='float64')
    
    
    Up0= nb_concatdot30(Wout, concat_nb0(np.sin(x), np.sin(x)**2))
    
    
    for i in nb.prange(0,N):
        for k in nb.prange(3):
            Jupred[k,i] = uin*W1[k,i]*np.cos(x[i]) + uin*2*W2[k,i]*np.sin(x[i])*np.cos(x[i])
        
    for i in nb.prange(0,N):
        for k in nb.prange(0,N):
            J[i,i] -= (K/N)*np.cos(x[k]-x[i]) 
        J[i,i] += K/N+F*np.cos(Up0[one_hot[i]]*uin-x[i])*(-1+ Jupred[one_hot[i],i])
                
    for i in nb.prange(0,N):
        for j in nb.prange(i+1,N):
            J[i,j] = (K/N)*np.cos(x[j]-x[i])
            J[j,i] = J[i,j]

            J[i,j] += F*np.cos(Up0[one_hot[i]]*uin-x[i])*(Jupred[one_hot[i],j])
            J[j,i] += F*np.cos(Up0[one_hot[j]]*uin-x[j])*(Jupred[one_hot[j],i])
            
    return J
    
@nb.njit()
def fkura_closed(t,x,p,pvec,Up,Wout,w,N):
    
    uin, K,F = p[0],p[1],p[2]
    one_hot = pvec
    
    y = np.empty(N,dtype='float64')
    R1=0
    R2=0
    
    Up0= nb_concatdot30(Wout, concat_nb0(np.sin(x), np.sin(x)**2))
    
    for i in nb.prange(N):
        R2 +=np.cos(x[i])
        R1 +=np.sin(x[i])   
    for i in nb.prange(N):
        y[i] = w[i] - K*R2*np.sin(x[i])/N + K*R1*np.cos(x[i])/N +  F* np.sin(Up0[one_hot[i]]*uin-x[i])
       
    return y

@nb.njit()
def concat_nb(x,y):
    res = np.empty( (x.shape[0]+y.shape[0],1), dtype = 'float64')
    for i in nb.prange(x.shape[0]+y.shape[0]):
        if i < x.shape[0]:
            res[i,0] = x[i,0]
        else:
            res[i,0] = y[i- x.shape[0],0 ]
    return res 

@nb.njit()
def concat_nb0(x,y):
    res = np.empty( (x.shape[0]+y.shape[0]), dtype = 'float64')
    for i in nb.prange(x.shape[0]+y.shape[0]):
        if i < x.shape[0]:
            res[i] = x[i]
        else:
            res[i] = y[i- x.shape[0] ]
    return res 

@nb.njit()
def numba_dot2(a, b):
    n = a.shape[0]
    dot = 0
    for i in range(n):
        dot += a[i] * b[i]
    return dot

@nb.njit()
def winding_row(xdiff):
    xtot =xdiff[np.where((xdiff >= -2) & (xdiff <2))].sum() - ((2*np.pi - xdiff[xdiff>2]  )).sum()  +  ((2*np.pi + xdiff[xdiff <-2])).sum()
    return xtot/(2*np.pi)

def unroll_parameter(p_dict):
    N = p_dict['N']
    K = p_dict['K']
    uin = p_dict['uin']
    h = p_dict['h']
    mu = p_dict['mu']
    sigma = p_dict['sigma']
    w = p_dict['w']
    F = p_dict['F']
    reg = p_dict['reg']
    one_hot = p_dict['one_hot']
    testlen = p_dict['test']
    trainlen = p_dict['train']
    initlen = p_dict['init']
    data_scale = p_dict['data_scale']
    return N,K,uin,h,mu,sigma,w,F,reg,one_hot,testlen,trainlen,initlen,data_scale


@nb.njit()
def solve_lya(N,m, x0, p,pvec, Wout,u0, w , ydata,dt,k=1,norm_freq =10):
    
    # based on MatLAB code by Anton O. Belyakov 
    # notable edits: remove transients, rk4 var computation
    
    #T : time vector 
    #X : [ x(t) , x(t+dt), x(t+2dt),....   ]
    
    LE_lya1dum =np.zeros(250)
    LE_lya2dum =np.zeros(250)
    LE_lya3dum =np.zeros(250)

    LEi_sort = []
    trJi_sort = []

    X = np.zeros((N,m)) 

    W1 = Wout[:,1:N+1] 
    W2 = Wout[:,N+1:]


    LE = np.zeros(k)  # Lyapunov exponents
    e = np.eye(N, k)  # make a matrix of variations (N x k)

    U_pred = np.zeros((3,201))
    U_pred[:,0] = u0

    trJ = 0  # integral of the trace of Jacobian matrix
    oldtraceJ = np.trace(Jkura_closed(0, x0,p,pvec,u0, W1,W2,Wout, w,N))  # trace of Jacobian matrix in previous step   
    Up=u0

    X[:,0] = x0

    k0=0 # indexing for first lya_dummy to compute mean and std

    for i in range(0,m-1):
        state= assemble(X[:,i], e, N,k)        
        X[:,i+1], e = disassemble(varRK4(dt*i, state, dt, fkura_closed, Jkura_closed,p, pvec,Up,W1,W2, Wout,w ,N,k),N,k)
        #print(e.shape)
        if i % norm_freq ==norm_freq-1: 
            e, nrm = gram_schmidt(e,k)
            LE = LE + np.log(nrm)


        xstate = concat_nb0(np.sin(X[:,i+1]), np.sin(X[:,i+1])**2)
        Up= nb_concatdot30(Wout, xstate)




        traceJ = np.trace(Jkura_closed(dt*(i+1), X[:,i+1],p,pvec,Up,W1,W2,Wout,w,N))
        trJ = trJ + 0.5 * dt * (oldtraceJ + traceJ)  # sum of LEs
        oldtraceJ = traceJ
        if i % norm_freq == norm_freq-1: 
            LEi_sort.append( LE / (dt*(i+1)))
            trJi_sort.append(trJ / (dt*(i+1)))
            LE_lya1dum[k0 % 250 ] = LEi_sort[-1][0]
            LE_lya2dum[k0 % 250 ] = LEi_sort[-1][1]
            LE_lya3dum[k0 % 250 ] = LEi_sort[-1][2]

            k0+=1

        if i<200:
            U_pred[:,i+1]= Up  
        elif i ==200:
            nmse_test = NMSE(U_pred[:,1:],ydata)
        elif numba_norm(nmse_test)>100 and i>10_000:
            print(f'NMSE - too large')
            break
        elif i>70_000:
            LEmean = np.array([ LE_lya1dum.mean(), LE_lya2dum.mean(), LE_lya3dum.mean()])
            LEstd = np.array([ LE_lya1dum.std(), LE_lya2dum.std(), LE_lya3dum.std()])


            if i>80_000 and np.all( np.abs(LEi_sort[-1]-LEi_sort[-2])<0.001) and np.all( np.abs(LEi_sort[-2]-LEi_sort[-3])<0.001) and np.all(np.abs(LEmean-LEi_sort[-1][:3])<0.0005) and np.all(LEstd<0.00025):  
                print(f'converged|steps:{i}')
                break

    LE_end = LE / (dt*(i+1)) # LEs calculation
    trJ_end = trJ / (dt*(i+1))  # calculation of LEs' sum


    
    return LE_end, trJ_end,LEi_sort, trJi_sort ,  U_pred,  i,nmse_test


def test_and_save(N,data_scaled,m, one_hot_output, initlen, trainlen,testlen, h,w,p,reg):

    K,F= p[1],p[2]

    ut0 =np.eye(3)[one_hot_output] @ data_scaled[:,::20]
    utmid =np.eye(3)[one_hot_output] @ data_scaled[:,10:][:,::20]
    ut1 =np.eye(3)[one_hot_output] @  data_scaled[:,20:][:,::20]

    x=whipeout(N,initlen, ut0,utmid,ut1, np.linspace(0,2*math.pi,N), h, w,p)
    X0= kura_run(N,trainlen, ut0[:,initlen:],utmid[:,initlen:],ut1[:,initlen:], x,h,w,p)

    X =  np.concatenate((np.expand_dims(np.ones(X0.shape[1]),axis=0),np.sin(X0),np.sin(X0)**2),axis=0)


    Yt = data_scaled [:,::20][:,(initlen+1):(initlen+trainlen+1)].T


    Wout = np.linalg.solve( np.dot(X,X.T) + reg*np.eye(2*N+1) , np.dot(X,Yt) ).T

    nmse_train = NMSE(  np.dot(Wout,X), Yt.T).mean()
    
    u0 = Yt.T[:,-1]
    x0= X0[:,-1] 
    

    ydata = data_scaled[:,::20][:, initlen+trainlen+1:initlen+trainlen+1+testlen]



    LE, trJ,LEi_sort, trJi_sort , U_pred,  conv_steps,nmse_test = solve_lya(N,m, x0,\
                                                                        p,one_hot_output, Wout,u0, w,ydata,\
                                                                        h,k=3,norm_freq=2)


    nmse_test_mean = np.round(nmse_test.mean(),4)

    shelf_id = f'K{K}|F{F}'

    
    
    #-------------------------------
    parameter_dict = dict()

    parameter_dict['K'] = K
    parameter_dict['F'] = F

    result_dict = dict()
    result_dict['NMSE_test_mean'] = nmse_test_mean
    result_dict['NMSE_test'] = nmse_test
    result_dict['NMSE_train'] = nmse_train
    result_dict['LEi_sort'] = LEi_sort
    result_dict['LE'] = LE
    result_dict['trJ'] =trJ
    result_dict['trJi_sort'] = trJi_sort
    result_dict['conv_steps'] = conv_steps

    #-------------------------------
    #print(f'N:{N}|K:{K}|{F}|h:{h}')
    print("NMSE train:", nmse_train )
    print('NMSE test: ',  nmse_test )
    print('LE: ',  LE )
    #-------------------------------
    with shelve.open('main/shelve/lya_low.shelve', 'c') as shelf:
        shelf[shelf_id] = [parameter_dict, result_dict]

def truncate(f, n):
    return np.floor(f * 10 ** n) / 10 ** n

def main():


    with shelve.open('main/data/data_sets.shelve', 'r') as shelf:
        data = shelf['lorenz'] 


    with shelve.open('shelve/param.shelve', 'r') as shelf:
        p_dict = shelf['lorenz']


    with shelve.open('main/data/low_grid.shelve', 'r') as shelf:
        Karray = shelf['K']
        Farray = shelf['F']

    N,K,uin,h,mu,sigma,w,F,reg,one_hot_output,testlen,trainlen,initlen, data_scale = unroll_parameter(p_dict)

    parameter_dict= dict()
    parameter_dict['reg'] = reg
    parameter_dict['one_hot'] = one_hot_output
    parameter_dict['test'] = testlen
    parameter_dict['train'] = trainlen
    parameter_dict['init'] = initlen
    parameter_dict['data_scale'] = data_scale 
    parameter_dict['uin'] = uin
    parameter_dict['h'] = h
    parameter_dict['mu'] = mu
    parameter_dict['sigma'] = sigma
    parameter_dict['w'] = w
    parameter_dict['N'] = N


    with shelve.open('main/shelve/lya_low.shelve', 'c') as shelf:
        shelf['parameters'] = parameter_dict

    m=100_000  
    data_scaled = data/data_scale
    testlen=200 

    s_step = 0

    for K,F in zip(Karray,Farray):    
        if K>15.5:
            print(f'F:{F}|K:{K}')
            start = time.time()
            p = np.array([uin, K,F])
            test_and_save(N,data_scaled,m, one_hot_output, initlen, trainlen,testlen, h, w,p,reg)
            print( f'duration: {time.time() - start}')
            print(f'step:{s_step}')    
                    
            s_step+=1




if __name__ == "__main__":
    main() 