import numpy as np
import matplotlib.pyplot as plt
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
    X = np.empty((N,runlen),dtype='float64')
    for j in nb.prange(runlen):
        U0 = ut0[:,j]
        Umid = utmid[:,j]
        U1 = ut1[:,j]
        
        x  = frk4_step(x0,uin*U0,uin*Umid,uin*U1,h,w,K,F,N)
        x0 = x
        X[:,j] = x
    return X

@nb.njit()
def create_u(one_hot , u):
    res = np.empty( (one_hot.shape[0],1),dtype = 'float64')
    for i in nb.prange(one_hot.shape[0]):
        if one_hot[i] ==0:
            res[i,0] = u[0]
        elif one_hot[i] ==1:
            res[i,0] = u[1]
        else:
            res[i,0] = u[2]
    return res

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
def numba_norm(a):
    n = a.shape[0]
    norm = 0
    for i in range(n):
        norm += a[i] * a[i]
    return np.sqrt(norm)


@nb.njit()
def varRK4(t, state, dt, f, p, pvec,Up,Wout,w,N):
    tmid = t + dt*0.5
    k1 = dt*fkura_closed(t,state,p,pvec,Up,Wout,w,N) 
    k2 = dt*fkura_closed(t,state + 0.5 * k1,p,pvec,Up,Wout,w,N)
    k3 = dt*fkura_closed(t,state + 0.5 * k2,p,pvec,Up,Wout,w,N) 
    k4 = dt*fkura_closed(t + dt ,state + k3,p,pvec,Up,Wout,w,N)
    y= (state + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4))
    return ( y % (2*math.pi))

    
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
def solve(N,m, x0, p,pvec, Wout,u0, w , ydata,dt,testlen=200):
    

    X = np.zeros((N,m+1),dtype='float64') 

    U_pred = np.zeros((3,testlen+1),dtype='float64')
    U_pred[:,0] = u0
    Up=u0

    X[:,0] = x0

    t=0
    for i in range(0,m):
        t=dt+t
        X[:,i+1] = varRK4(t, X[:,i], dt, fkura_closed, p, pvec,Up,Wout,w,N)
        xstate = concat_nb0(np.sin(X[:,i+1]), np.sin(X[:,i+1])**2)
        Up= nb_concatdot30(Wout, xstate)

        if i<200:
            U_pred[:,i+1]= Up  
        elif i ==200:
            nmse_test = NMSE(U_pred[:,1:],ydata)
    winding_total = np.array([np.floor(np.abs(winding_row(x))) for x in ((X[:,1:i]-X[:,:i-1]))])

    return winding_total , nmse_test


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
    

    ydata = data_scaled[:,::20][:, initlen+trainlen+1:initlen+trainlen+testlen+1]

    winding_total , nmse_test = solve(N,m, x0, p,one_hot_output, Wout,u0, w , ydata,h)
 
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
    result_dict['winding'] = winding_total
    #-------------------------------
    #print(f'N:{N}|K:{K}|{F}|h:{h}')
    print("NMSE train:", nmse_train )
    print('NMSE test: ',  nmse_test )
    
    #-------------------------------
    with shelve.open('shelve/nmse_low.shelve', 'c') as shelf:
        shelf[shelf_id] = [parameter_dict, result_dict]



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


    with shelve.open('main/shelve/nmse_low.shelve', 'c') as shelf:
        shelf['parameters'] = parameter_dict


    m=20_000    #80_000
    data_scaled = data/data_scale
    testlen=200 

    s_step = 0


    for K,F in zip(Karray,Farray):
    

        print(f'F:{F}|K:{K}')
        start = time.time()
        p = np.array([uin, K,F])
        test_and_save(N,data_scaled,m, one_hot_output, initlen, trainlen,testlen, h, w,p,reg)
        print( f'duration: {time.time() - start}')
        print(f'step:{s_step}')    
        s_step+=1



if __name__ == "__main__":
    main() 