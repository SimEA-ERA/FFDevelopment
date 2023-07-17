"""
Created on Fri Mar 19 18:49:40 2021

@author: nikolas
This is a library to develop FF. 
Currently it supports FF development for physisorption of small molecules to surfaces"
"""
from numba import jit,prange
import os.path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import minimize, differential_evolution,dual_annealing,direct,brute
from time import perf_counter
import coloredlogs
import logging
import itertools
from scipy.interpolate import BSpline
import collections
import six

class logs():
    '''
    A class for modifying the loggers
    '''
    def __init__(self):
        self.logger = self.get_logger()
        
    def get_logger(self):
    
        LOGGING_LEVEL = logging.CRITICAL
        
        logger = logging.getLogger(__name__)
        logger.setLevel(LOGGING_LEVEL)
        logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
        formatter = logging.Formatter(logFormat)
        

        if not logger.hasHandlers():
            logfile_handler = logging.FileHandler('FF_develop.log',mode='w')
            logfile_handler.setFormatter(formatter)
            logger.addHandler(logfile_handler)
            self.log_file = logfile_handler
            stream = logging.StreamHandler()
            stream.setLevel(LOGGING_LEVEL)
            stream.setFormatter(formatter)
            
            logger.addHandler(stream)
         
        fieldstyle = {'asctime': {'color': 'magenta'},
                      'levelname': {'bold': True, 'color': 'green'},
                      'filename':{'color':'green'},
                      'funcName':{'color':'green'},
                      'lineno':{'color':'green'}}
                                           
        levelstyles = {'critical': {'bold': True, 'color': 'red'},
                       'debug': {'color': 'blue'}, 
                       'error': {'color': 'red'}, 
                       'info': {'color':'cyan'},
                       'warning': {'color': 'yellow'}}
        
        coloredlogs.install(level=LOGGING_LEVEL,
                            logger=logger,
                            fmt=logFormat,
                            datefmt='%H:%M:%S',
                            field_styles=fieldstyle,
                            level_styles=levelstyles)
        return logger
    
    def __del__(self):
        try:
            self.logger.handlers.clear()
        except:
            pass
        try:    
            self.log_file.close()
        except:
            pass
        return
logobj = logs()        
logger = logobj.logger
def try_beebbeeb():
    try:
        import winsound
        winsound.Beep(500, 1000)
        import time
        time.sleep(0.5)
        winsound.Beep(500, 1000)
    except:
        pass
    return

class maps:
    def __init__():
        pass
def get_colorbrewer_colors(n):
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
    nc = len(colors)
    ni = int(nc/n)
    colors = [colors[i] for i in range(0,nc,ni)]
    return colors
import collections
def iterable(arg):
    return (
        isinstance(arg, collections.Iterable) 
        and not isinstance(arg, six.string_types)
    )

def make_dir(path):
    try:
        if not os.path.exists(path):
            x = os.system('mkdir ' + path)
            if x != 0:
                raise ValueError
        else:
            return
    except:
        path = '\\'.join(path.split('/'))
        make_dir(path)
    else:
        print('Created DIR: '+path)
    return


def weighting(E,weighting_method,T=1,w=1):
    E = np.array(E)
    if weighting_method =='constant':    
        weights = np.ones(len(E))*w
    elif weighting_method == 'Boltzman':
        weights = w*np.exp(-E/T)/np.sum(np.exp(-E/T))
        weights /=weights.mean() 
    elif weighting_method == 'linear':
        weights = w*(E.max()-E+1.0)/(E.max()-E.min()+1.0)
    elif weighting_method =='divbyEsquare':
        weights = w*1.0/(E**2+1)
    elif weighting_method =='linear+divbyEsquare':
        weights = w*1.0/(E**2+stability_value)
        weights *= (E.max()-E+1.0)/(E.max()-E.min()+stability_value)
    elif weighting_method == 'Boltzman+divbyEsquare':
        weights = w*np.exp(-E/T)/(np.sum(np.exp(-E/T))*(E**2+stability_value))
    return weights

@jit(nopython=True,fastmath=True)
def calc_dist(r1,r2):
    r = r2 - r1
    d = np.sqrt(np.dot(r,r))
    return d

@jit(nopython=True,fastmath=True)
def calc_angle(r1,r2,r3):
    d1 = r1 -r2 ; d2 = r3-r2
    nd1 = np.sqrt(np.dot(d1,d1))
    nd2 = np.sqrt(np.dot(d2,d2))
    cos_th = np.dot(d1,d2)/(nd1*nd2)
    return np.arccos(cos_th)

@jit(nopython=True,fastmath=True)
def calc_dihedral(r1,r2,r3,r4):
    d1 = r2-r1
    d2 = r3-r2
    d3 = r4-r3
    c1 = np.cross(d1,d2)
    c2 = np.cross(d2,d3)
    n1 = c1/np.sqrt(np.dot(c1,c1))
    n2 = c2/np.sqrt(np.dot(c2,c2))
    m1= np.cross(n1,d2/np.sqrt(np.dot(d2,d2)))
    x= np.dot(n1,n2)
    y= np.dot(m1,n2)
    dihedral = np.arctan2(y, x)
    return dihedral
        
    

@jit(nopython=True,fastmath=True)
def gauss(x,mu=0,sigma=1):
    px = np.exp(-0.5*( (x-mu)/sigma)**2)/(np.sqrt(2*np.pi*sigma**2))
    return px

def most_min(arr,m):
    return arr.argsort()[0:m]

@jit(nopython=True,fastmath=True)
def norm2squared(r1):
    r = np.dot(r1,r1)
    return r

@jit(nopython=True,fastmath=True)
def norm2(r1):
    return np.sqrt(np.dot(r1,r1))

@jit(nopython=True,fastmath=True)
def norm1(r1):
    r = np.sum(np.abs(r1))
    return r

@jit(nopython=True,fastmath=True)
def u_Alvarez(r,sigma,q):
    u = (q/r)*(1-np.sign(q)*(sigma/r)**9)
    return u

@jit(nopython=True,fastmath=True)
def du_Alvarez(r,sigma,q):
    du = -sigma**8/r**10 - q/r**2
    return du
def Atridag(n):
    A =np.zeros((n,n))
    A[0,0]=1 ; A[n-1,n-1]=1
    for i in range(1,n-1):
        A[i,i]=4
        A[i,i+1]=1
        A[i,i-1]=1#
    return A

@jit(nopython=True,fastmath=True)
def u_inverse_double_exp(r,A,B,re,C,D,rl):

    u = A*np.exp(B*(r-re)) - C*np.exp(D*(r-rl))
    return u
def u_polyexp(rhos,*knots):
    u = 0
    #irhos = 1/(1+rhos)
    #irhos = rhos
    n = len(knots)
    n2 = int(n/2)
    rh2 = 1/rhos
    for i in range(n2):
        c = knots[i]
        a = knots[n2+i]
        u+= c*np.exp(-a*rh2)
    return u 

@jit(nopython=True,fastmath=True)
def u_flex(rhos,c1,c2,a,b,re):
    u = de*(1-np.exp(b*rhos)+rhos*np.exp(-a*(rhos-re)))
    return u

@jit(nopython=True,fastmath=True)
def u_iexponent(rhos,x):
    n = x.size
    u = np.empty_like(rhos)
    n2  = int(n/2)
    for j in range(u.size): 
        u[j]=0.0
    #di = 1/float(n2)
    for i in range(n2):
        coeff = x[i*n2]
        ex = x[i*n2+1]
        u+= coeff*(1-np.exp(ex*rhos))
    #u+=np.exp(-x[-2]*(rhos-x[-1]))
    return u

@jit(nopython=True,fastmath=True)
def u_exponent(rhos,x):
    n = x.size
    u = np.empty_like(rhos)
    
    for j in range(u.size): 
        u[j]=0.0
    di = 1/float(n-1)
    for i in range(n-1):
        coeff = x[i]
        ex = di*i 
        u+= coeff*(1-np.exp(ex*(rhos-x[-1]*i)))
    #u+=np.exp(-x[-2]*(rhos-x[-1]))
    return u


@jit(nopython=True,fastmath=True)
def u_polynomial(rhos,x):
    n = x.size
    u = np.empty_like(rhos)
    
    for j in range(u.size): 
        u[j] = 0.0
    
    n2 =np.uint(n/2)
    for i in range(n2):
        coeff = x[i]
        ex = x[i+n2]
        u+= coeff*rhos**ex
    return u

@jit(nopython=True,fastmath=True)
def numba_factorial(n):
    if n==0:
        return 1
    f = 1
    for i in range(1,n+1):
        f *= i
    return f

@jit(nopython=True,fastmath=True)
def numba_bezier_rt(t,r,M):
    n = r.size
    x=0.0
    for i in range(n):
        xi = r[i]
        #temp = 0.0
        for j in range(i,n):
            x += xi*M[i,j]*(t**j)
        #x+=temp*xi
    return x

@jit(nopython=True,fastmath=True)
def numba_bezier_drdt(t,r,M):
    n = r.size
    x = 0.0
    for i in range(n):
        xi = r[i]
        for j in range(i,n):
            x+=xi*float(j)*M[i,j]*t**(j-1)
    return x

@jit(nopython=True,fastmath=True)
def numba_bezier_rtdrdt_ratio(t,r,M,xval):
    n = r.size
    dx = 0.0
    x=0.0
    for i in range(n):
        xi = r[i]
        for j in range(i,n):
            mij = M[i,j]
            tj = t**(j-1)
            dx+=xi*float(j)*mij*tj
            x +=xi*mij*t*tj
    return (x-xval)/dx

@jit(nopython=True,fastmath=True)
def numba_find_bezier_t__bisect(xval,r,M):
    tol = 1e-4
    n = r.size
    a=0
    b=1.0
    told = a 
    res = 1
    #it = 0
    while res > tol:
        tnew = 0.5*(a+b)
        xnew = numba_bezier_rt(tnew,r,M)
        if xnew > xval:
            b = tnew
        else:
            a = tnew
        #print(it,a,b,tnew)
        res = np.abs(xnew-xval) 
        #it+=1
        #if it >100:
            #return 0
    return tnew


@jit(nopython=True,fastmath=True)
def numba_find_bezier_t__newton(xval,r,M,tguess):
    tol = 1e-6
    res =1.0
    n = r.size
    #tguess = 0.5
    told = tguess
    tnew = tguess
    while res > tol:
        
        #ft = numba_bezier_rt(tnew,r,M) - xval
        #fdt = numba_bezier_drdt(tnew,r,M)
        #tnew  = told - ft/fdt
        tnew = told - numba_bezier_rtdrdt_ratio(tnew,r,M,xval)
        
        res = np.abs(tnew-told)
        told = tnew

    return tnew

@jit(nopython=True,fastmath=True,parallel=True)
def numba_find_bezier_taus(xvals,rho_max,r,M):
    taus = np.empty_like(xvals)
    tguess=0.5
    for i in prange(xvals.size):
        xv = xvals[i]
        if xv==0:
            taus[i] = 0.0
        elif xv<rho_max:
            taus[i] = numba_find_bezier_t__newton(xv,r,M,tguess)
        else:
            taus[i] = 1.0
        #taus[i] = numba_find_bezier_t__bisect(xv,r,M)
    return taus

@jit(nopython=True,fastmath=True)
def numba_bezier_matrix_coef(i,j,N):
    s = (-1)**(j-i)
    nj = numba_combinations(N, j)
    ij = numba_combinations(j,i)
    mij = s*nj*ij
    return mij

@jit(nopython=True,fastmath=True)
def numba_bezier_matrix(N):
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):      
            M[i,j] = numba_bezier_matrix_coef(i,j,N-1)
    return M

@jit(nopython=True,fastmath=True)
def numba_combinations(n,r):
    a = 1
    for i in range(r+1,n+1):
        a*=i
    b = numba_factorial(n-r)
    return float(a/b)


@jit(nopython=True,fastmath=True)
def numba_bezier_rtaus(taus,r,M):
    n = r.size
    yr = np.zeros((taus.size,))
    for i in range(n):
        xi = r[i]
        for j in range(i,n):
            yr += xi*M[i,j]*taus**j
    return yr

@jit(nopython=True,fastmath=True)
def xy_bezier(x,y,taus):
    
    ny = y.shape[0]
    M = numba_bezier_matrix(ny)
    xu = numba_bezier_rtaus(taus,x,M)
    yu = numba_bezier_rtaus(taus,y,M)
    return xu,yu

@jit(nopython=True,fastmath=True)
def u_bezierXY(rhos,bxy):

    nbs = int(bxy.shape[0]/2)
    bx = np.zeros((nbs,))
    by = np.zeros((nbs,))
    for i in range(nbs):
        bx[i] = bxy[i]
        by[i] = bxy[nbs+i]
    
    M = numba_bezier_matrix(nbs)

    taus = numba_find_bezier_taus(rhos,rho_max,bx,M)

    u = numba_bezier_rtaus(taus,by,M)
    
    return u

@jit(nopython=True,fastmath=True)
def u_bezier(rhos,yp):
    
    rho_max = yp[0]
    y = yp[1:]
    
    ny = y.shape[0] 
    dx = rho_max/float(ny-1)
    x = np.empty_like(y)
    
    M = numba_bezier_matrix(ny)
    
    for i in range(ny):
        x[i] = float(i)*dx

    taus = numba_find_bezier_taus(rhos,rho_max,x,M)

    u = numba_bezier_rtaus(taus,y,M)
    
    return u

@jit(nopython=True,fastmath=True)
def u_bspline(rhos,x):
    k = 3
    t = range(x.size+k)
    fu = BSpline(t,x,k)
    return fu(rhos)




@jit(nopython=True,fastmath=True)
def u_LJnm(r,x):
    r0 = x[0]
    e0 = x[1]
    n = x[2]
    m = x[3]
    ulj = e0*( m*(r0/r)**n - n*(r0/r)**m )/(n-m)
    return ulj 

@jit(nopython=True,fastmath=True)
def u_LJnmq(r,e0,r0,n,m,q):
    ulj = e0*( m*(r0/r)**n - n*(r0/r)**m )/(n-m) -q/r
    return ulj 

def du_LJnm(r,sigma,epsilon,n,m):
    dulj = e0*(m*(-n/r)*(r0/r)**n - n*(-m/r)*(r0/r)**m )/(n-m)
    return dulj

def du_compass(r,r0,epsilon,q):
    ulj = epsilon*(18/r)*((r0/r)**6 - (r0/r)**6 ) + c/r**2
    return ulj 

@jit(nopython=True,fastmath=True)
def u_compass(r,r0,epsilon,c):
    ulj = epsilon*( 2*(r0/r)**9 - 3*(r0/r)**6 ) - c/r
    return ulj 

@jit(nopython=True,fastmath=True)
def u_qeff(r,x):
    qeff = x[0]*627.503
    u = -qeff/r
    return 

@jit(nopython=True,fastmath=True)
def u_LJq(r,x):
    sigma = x[0]
    epsilon = x[1]
    qeff = x[2]
    ulj = 4.0*epsilon*( (sigma/r)**12 - (sigma/r)**6 )-qeff*627.503/r
    return ulj 

@jit(nopython=True,fastmath=True)
def u_LJ(r,x):
    sigma = x[0]
    epsilon = x[1]
    ulj = 4.0*epsilon*( (sigma/r)**12 - (sigma/r)**6 )
    return ulj 

def du_LJ(r,sigma,epsilon,p1=12,p2=6):
    dulj = 4.0*epsilon*((-12.0/r)*(sigma/r)**12 - (-6.0/r)*(sigma/r)**6 )
    return dulj

@jit(nopython=True,fastmath=True)
def u_exp_vdwq(r,A,B,re,c1,c6):
    r6 = r**6
    u = A*np.exp(-B*(r-re)) - c1/r - c6**6/r6
    return u

@jit(nopython=True,fastmath=True)
def u_double_exp(r,x):
    A = x[0]
    B = x[1]
    re = x[2]
    C = x[3]
    D = x[4]
    rl = x[5]
    u = A*np.exp(-B*(r-re)) - C*np.exp(-D*(r-rl))
    return u

def du_double_exp(r,A,B,re,C,D,rl):

    u = -B*A*np.exp(-B*(r-re)) + D*C*np.exp(-D*(r-rl))
    return u

@jit(nopython=True,fastmath=True)
def u_exp_poly(r,A,B,re,c1,c2,c3,c4,c5,c6):
    r2 = r*r 
    r3 = r2*r 
    r4 = r3*r 
    r5 = r4*r
    r6 = r5*r
    u = A*np.exp(-B*(r-re)) - c1/r - c2**2/r2 - c3**3/r3 -c4**4/r4 - c5**5/r5 - c6**6/r6
    return u

def du_exp_poly(r,A,B,re,c1,c2,c3,c4,c5,c6):
    r = r*r
    r2 = r*r 
    r3 = r2*r 
    r4 = r3*r 
    r5 = r4*r
    r6 = r5*r
    du = -A*B*np.exp(-B*(r-re)) + c1/r + 2*c2**2/r2 + 3*c3**3/r3 +4*c4**4/r4 + 5*c5**5/r5 + 6*c6**6/r6
    return u

@jit(nopython=True,fastmath=True)
def u_poly12(r,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12):
    r2 = r*r 
    r3 = r2*r 
    r4 = r3*r 
    r5 = r4*r
    r6 = r5*r
    r7 = r6*r ; r8 = r7*r ; r9=r8*r ; 
    r10=r9*r ; r11 = r10*r ; r12= r11*r
    u = c12**12/r12 + c11**11/r11 + c10**10/r10 + c9**9/r9 + c8**8/r8 + c7**7/r7 \
    - c1/r - c2**2/r2 - c3**3/r3 -c4**4/r4 - c5**5/r5 - c6**6/r6
    return u

@jit(nopython=True,fastmath=True)
def du_poly12(r,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12):
    r=r*r
    r2 = r*r 
    r3 = r2*r 
    r4 = r3*r 
    r5 = r4*r
    r6 = r5*r
    r7 = r6*r ; r8 = r7*r ; r9=r8*r ; 
    r10=r9*r ; r11 = r10*r ; r12= r11*r
    du = -12*c12**12/r12 -11*c11**11/r11 -10*c10**10/r10-9*c9**9/r9 - 8*c8**8/r8 - 7*c7**7/r7 \
    + c1/r + 2*c2**2/r2 + 3*c3**3/r3 +4*c4**4/r4 + 5*c5**5/r5 + 6*c6**6/r6
    return du



@jit(nopython=True,fastmath=True)
def u_inverseMorse(r,x):
    #r = 1/(r+1e-16)
    re = x[0]
    De = x[1]
    alpha = x[2]
    t1 = alpha*(r-re)
    u = De*(np.exp(2.0*t1)-2.0*np.exp(t1) - (np.exp(-2*alpha*re)-2*np.exp(-alpha*re)))
    return u



@jit(nopython=True,fastmath=True)
def u_Morse(r,x):
    re = x[0]
    De = x[1]
    alpha = x[2]
    t1 = -alpha*(r-re)
    u = De*(np.exp(2.0*t1)-2.0*np.exp(t1))
    return u

@jit(nopython=True,fastmath=True)
def u_Morseq(r,x):
    re = x[0]
    De = x[1]
    alpha = x[2]
    c = x[3]
    t1 = -alpha*(r-re)
    u = De*(np.exp(2.0*t1)-2.0*np.exp(t1))-(c/r)**6
    return u

@jit(nopython=True,fastmath=True)
def u_Buck(r,A,B,C):
    u = A*np.exp(-B*r) - C/r**6
    return u

def du_Buck(r,A,B,C):
    du =-A*B*np.exp(-B*r) + 6.0*C/r**7
    return du
def du_Morse(r,re,De,alpha):
    t1 = -alpha*(r-re)
    du = De*(-2.0*alpha*np.exp(2.0*t1)+2.0*alpha*np.exp(t1))
    return du

def du_Morse(r,re,De,alpha,c):
    t1 = -alpha*(r-re)
    du = De*(-2.0*alpha*np.exp(2.0*t1)+2.0*alpha*np.exp(t1)) +c/r**2
    return du

def u_Buck_v(r,A,B,C):
    u =np.exp(-B*r+A)-(C/r)**6
    return u
def du_Buck_v(r,A,B,C):
    du = -B*np.exp(-B*r+A)+6*(C**6)/r**7
    return du


def Uphi_improper( phi,k=167.4*0.239006,phi0=np.pi):
    u = 0.5*k*(phi-phi0)**2
    return u

def Uang(theta,kang,th0):
    u = 0.5*kang*(theta-th0)**2
    return u

def Ubond(r,kb,r0):
    u = 0.5*kb*(r-r0)**2
    return u


                
class Setup_Interfacial_Optimization():
    
    defaults = {
        'representation':'AA',
        'PW_model':'LJ',
        'storing_path':'FFresults',
        'run': '0',
        'dpi':300,
        'figsize':(3.5,3.5),
        'runpath_attributes':['PW_model'],
        'uncertainty_method':'no',
        'max_ener':1.0,
        'max_force':0.003,
        'split_method':'random',
        'colsplit':'label',
        'train_colvars': ['',],
        'optimization_method':'DA',
        'opt_disp':False,
        'maxiter':30,
        'maxiter_i':50,
        'tolerance':1e-4,
        'sampling_method':'random',
        'seed':1291412,
        'train_perc':0.8,
        'validation_set':'colname1:colv1 , colv2, colv3 & colname2: colv1 , colv2',
        'regularization_method': 'ridge',
        'reg_par': 0.0,
        'polish':False,
        'popsize':30,
        'mutation':(0.5,1.0),
        'recombination':0.7,
        'initial_temp':5230.0,
        'restart_temp_ratio':2e-5,
        'local_search_scale':1.0,
        'accept':-5.0,
        'visit':2.62,
        'weighting_method':'constant',
        'w':10.0,
        'bT':100.0,
        'costf':'MSE',
        'nLD':0,
        'nPW':1,
        'LD_model':'polynomial',
        'LD_types' : [''],
        'rho_r0' : [1.1],
        'rho_rc': [6.0],
        }
    
    def __init__(self,fname):
        '''
        A Constructor of the setup of Interfacial Optimization
        
        Parameters
        ----------
        fname : string
            File to read the setup.
        Raises
        ------
        Exception
            If you initialize wrongly the parameters.
            
        Returns
        -------
        None.
        
        '''
        
        print('setting algorithm from file "{:s}"'.format(fname))
        def my_setattr(self,attrname,val,defaults):
            if attrname not in defaults:
                raise Exception('Uknown input variable "{:s}"'.format(attrname))
            ty = type(defaults[attrname])
            if ty is list or ty is tuple:
                
                tyi = type(defaults[attrname][0])
                attr = ty([tyi(v) for v in val])
            else:
                attr = ty(val)
            setattr(self,attrname,attr)

            return
        # Defaults

        
        defaults = self.defaults
        
        with open(fname,'r') as f:
            lines = f.readlines() 
            f.closed
        
        #strip comments
        for j,line in enumerate(lines):
            for i,s in enumerate(line):
                if '#' == s: 
                    lines[j]=lines[j][:i]+'\n'
                lines[j] = lines[j].strip()
                
        section_lines = dict()
        for j,line in enumerate(lines):
            if '&' in line:
                section_lines[line.split('&')[-1].strip()] = j
        
        #get_attributes
        for j,line in enumerate(lines):
            
            if '&' in line:
                break 
            
            if '=' in line:
                li = line.split('=')
                var = li[0].strip() ; value = li[1].strip()
                if value.isdigit():
                    value = int(value)
                elif value.replace('.','',1).isdigit():
                    value = float(value)
            elif ':' in line:
                li = line.split(':')
                var = li[0].strip() ; value = [] 
                for x in li[1].split():
                    if x.isdigit(): y = int(x)
                    elif x.replace('.','',1).isdigit(): y = float(x)
                    else: y = x
                    value.append(y)
            else:
                continue
   
            my_setattr(self,var,value,defaults)
            
        for atname,de in defaults.items():
                if not hasattr(self,atname):
                    setattr(self,atname,de)
        #Get initial conditions
        
        for prefix in ['PW','LD']:
            model = prefix +'_model' 
            n = getattr(self,'n'+prefix)
            for i in range (n):
                key = '{:s}{:d}'.format(prefix,i)
                params = self.get_params(lines[section_lines[key]:],prefix)
                attrname = 'init' +key
                setattr(self,attrname,params)
                self.initExceptions(model,params)
        
        return 
    


        return
    def initExceptions(self,model,params):
        if model =='bezierXY':
            modelpars = self.model_parameters(model, params.columns)
            bx = [x for x in modelpars if 'bx' in x]
            for i in range(1,len(bx)):
                bo = bx[i-1]
                bn = bx[i]
                for j,pars in params.iterrows():
                    if pars[bn]<pars[bo]:
                        raise ValueError('Initialization Error/non TYPE: {} {} must be greater than {}'.format(j,bn,bo))
                    if pars[bn+'_b'][0]<pars[bo+'_b'][1]:
                        raise ValueError('Initialization Error/non TYPE:  {}  lower bound {} must be greater than the upper bound {}'.format(j,bn,bo))
                    if pars[bn]<pars[bo+'_b'][1]:
                        raise ValueError('Initialization Error/non TYPE:  {} {} must be greater than the upper bound {}'.format(j,bn,bo))
                    if pars[bn+'_b'][0]<pars[bo]:
                        raise ValueError('Initialization Error/non TYPE:  {}  lower bound {} must be greater than {}'.format(j,bn,bo))
        return
    
    def get_params(self,lines,prefix):
        
        types = []
        params = dict()
        for j,line in enumerate(lines):
            
            if 'TYPE' in line:
                inter = line.split('TYPE')[1].split()
                if prefix not in ['LD',]:
                    inter = self.sort_type(inter)
                else:
                    inter = tuple(inter)
                types.append(inter)
            if ':' in line:
                li = line.split(':')
                var = li[0].strip() ; value = [float(x) for x in li[1].split()]
                if len(value) != 4:
                    raise Exception('In line "{:s}..." Cannot determine parameters. Expected 4 values {:1d} were given " \n Give "value optimize_or_not low_bound upper_bound'.format(line[0:30],len(value)))
                if var not in params.keys():
                    params[var] = []; params[var+'_opt'] = [] ; params[var+'_b'] = [] 
                params[var].append(value[0])
                params[var+'_opt'].append(bool(value[1]))
                params[var+'_b'].append((value[2],value[3]))
            if '/' in line:
                break
            
        params['types'] = types
        params = pd.DataFrame(params,index=params['types']).drop('types',axis=1)
        
        return params
    
    def model_parameters(self,model,cols):
        if model =='Morse' or model=='inverseMorse':
            return ['re','De','alpha']
        elif model in ['Morseb','inverseMorseb']:
            return ['re','De','alpha','beta']
        elif model=='Buck':
            return ['A','B','C']
        elif model =='LJ':
            return ['sigma','epsilon']
        elif model =='LJq':
            return ['sigma','epsilon','qeff']
        elif model =='LJnm':
            return ['r0','e0','n','m']
        elif model=='LJnmq':
            return ['r0','e0','n','m','q']
        elif model =='Morseq':
            return ['re','De','alpha','c']
        elif model =='exp_poly':
            return ['A','B','re','c1','c2','c3','c4','c5','c6']
        elif model =='poly12':
            return ['c{:d}'.format(c) for c in range(1,13)]
        elif model =='exp_vdwq' or model == 'inverse_exp_vdwq':
            return ['A','B','re','c1','c6']
        elif model =='double_exp' or model =='inverse_double_exp':
            return ['A','B','re','C','D','rl']
        elif model=='compass':
            return ['r0','epsilon','c']
        elif model in ['bspline','polynomial','exponent','iexponent','bezier']:
            return [c for c in cols if '_' not in c ]
            return [c for c in cols if '_' not in c ]
        elif model in [ 'bezierXY']:
            bx = [c for c in cols if '_' not in c and 'bx' in c]
            by = [c for c in cols if '_' not in c and 'by' in c]
            bx.extend(by)
            b = bx
            return b
        else:
            raise Exception('model {:s} is not implemented'.format(model))
    
    @property 
    def model_function(self):
        try:
            return globals()['u_'+self.PW_model]
        except:
            raise NotImplemented('u_'+self.PW_model)
    
    @property 
    def model_gradient_function(self):
        try:
            return globals()['du_'+self.PW_model]
        except:
            raise NotImplemented('du_'+self.PW_model)
            
    
    def number_of_model_parameters(self,model,cols):
        return len(self.model_parameters(model,cols))
    
    @staticmethod
    def sort_type(t):
        if len(t) ==2:
            return tuple(np.sort(t))
        elif len(t)==3:
            if t[0]<=t[2]: 
                ty = tuple(t)
            else:
                ty = t[2],t[1],t[0]
            return ty
        elif len(t)==4:
            if t[0]<=t[3]: 
                ty = tuple(t)
            else:
                ty = t[3],t[2],t[1],t[0]
            return ty
        else:
            return NotImplemented
    
    @property                
    def runpath(self):
        r = self.storing_path
        for a in self.runpath_attributes:
            r = os.path.join(r,str(getattr(self,a)))
        return r
    

    def write_running_output(self,pandas_params = {}):

        def type_var(v,ti,s):
            if ti is int: s+='{:d} '.format(v)
            elif ti is str: s+='{:s} '.format(v)
            elif ti is float: s+='{:8.6f} '.format(v)
            elif ti is bool:
                if v: s+='1 ' 
                else: s+='0 '
            return s

        def write(file,name,var):
            s = '{:15s}'.format(name)
            t = type(var)

            if t is list or t is tuple:
                ti = type(var[0])
                s +=' : '
                for v in var:
                    s = type_var(v,ti,s)
            else:
                s+= ' = '
                s = type_var(var,t,s)
            s+='\n'
            
            file.write(s)
            return
        
        fname = '{:s}/runned.in'.format(self.runpath)
        with open(fname,'w') as file:
            add_empty_line = [6,12,27,31] 
            
            for i,(k,v) in enumerate(self.defaults.items()):
                var = getattr(self,k)
                write(file,k,var)
                if i in add_empty_line: 
                    file.write('\n')
            file.write('\n')
            
            for k,df in pandas_params.items():
                file.write('&{:s}\n'.format(k))
                parnames = [ c for c in df.columns if '_' not in c]
                for j,d in df.iterrows():
                    file.write('TYPE '+' '.join(j) +'\n')
                    for p in parnames:
                        file.write('{:10s} : {:8.6f}  {:d}  {:8.6f}  {:8.6f}  \n'.format(
                                    p, d[p], int(d[p+'_opt']), d[p+'_b'][0], d[p+'_b'][1]))
                
                file.write('/\n\n')
                
            file.closed
        return
                    
    def __repr__(self):
        x = 'Attribute : value \n--------------------\n'
        for k,v in self.__dict__.items():
            x+='{} : {} \n'.format(k,v)
        x+='--------------------\n'
        return x

    

class Interactions():
    
    def __init__(self,data,atom_model = 'AA',
            vdw_bond_dist=4, find_vdw_unconnected = False,
            find_bonds=False, find_vdw_connected=False,
            find_dihedrals=False,find_angles=False,find_densities=False,
            excludedBondtypes=[],**kwargs):
        
        self.data = data
        self.atom_model = atom_model
        self.vdw_bond_dist = vdw_bond_dist
        self.find_bonds = find_bonds
        self.find_vdw_unconnected = find_vdw_unconnected
        self.find_vdw_connected = find_vdw_connected
        self.find_angles = find_angles
        self.find_dihedrals = find_dihedrals
        self.find_densities = find_densities
        if find_densities:
            for rh in ['rho_r0','rho_rc']:
                if rh  not in kwargs:
                    raise Exception('{:s} is not given'.format(rho_r0))
            if kwargs['rho_r0'] >= kwargs['rho_rc']:
                raise Exception('rho_rc must be greater than rho_r0')
        
        for i in range(len(excludedBondtypes)):
            excludedBondtypes.append((excludedBondtypes[i][1],excludedBondtypes[i][0]))
        
        self.excludedBondtypes = excludedBondtypes
        for k,v in kwargs.items():
            setattr(self,k,v)
        return
    
    @staticmethod
    def bonds_to_python(Bonds):
        if len(Bonds) == 0:
            return Bonds
        bonds = np.array(Bonds,dtype=int)
        min_id = bonds[:,0:2].min()
        logger.debug('min_id = {:d}'.format(min_id))
        if  min_id == 1:
            bonds[:,0:2]-=1
        elif min_id == 0:
            logger.info('The ids start from 0')
        else:
            error = 'Error: The ids start from {:d} and I dont know ho to handle this'.format(min_id)
            logger.debug(error)
            raise Exception(error)
        return bonds
    
    @staticmethod
    def get_connectivity(bonds,types,excludedBondtypes):
        conn = dict()     
        for b in bonds:
            i,t = Interactions.sorted_id_and_type(types,(b[0],b[1]))
            if t in excludedBondtypes:
                continue
            conn[i] = t
        return conn
    
    @staticmethod
    def get_angles(connectivity,neibs,types):
        '''
        Computes the angles of a system in dictionary format
        key: (atom_id1,atom_id2,atom_id3)
        value: object Angle
        Method:
            We search the neihbours of bonded atoms.
            If another atom is bonded to one of them an angle is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        angles = dict()
        for k in connectivity.keys():
            #"left" side angles k[0]
            for neib in neibs[k[0]]:
                if neib in k: continue
                ang_id ,ang_type = Interactions.sorted_id_and_type(types,(neib,k[0],k[1]))
                if ang_id[::-1] not in angles.keys():
                    angles[ang_id] = ang_type
            #"right" side angles k[1]
            for neib in neibs[k[1]]:
                if neib in k: continue
                ang_id ,ang_type = Interactions.sorted_id_and_type(types,(k[0],k[1],neib))
                if ang_id[::-1] not in angles.keys():
                    angles[ang_id] = ang_type  
        tf = perf_counter()
        logger.info('angles time --> {:.3e} sec'.format(tf-t0))
        return angles
    
    @staticmethod
    def get_dihedrals(angles,neibs,types):
        '''
        Computes dihedrals of a system based on angles in dictionary
        key: (atom_id1,atom_id2,atom_id3,atom_id4)
        value: object Dihedral
        Method:
            We search the neihbours of atoms at the edjes of Angles.
            If another atom is bonded to one of them a Dihedral is formed is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        dihedrals=dict()
        for k in angles.keys():
            #"left" side dihedrals k[0]
            for neib in neibs[k[0]]:
                if neib in k: continue
                dih_id,dih_type = Interactions.sorted_id_and_type(types,(neib,k[0],k[1],k[2]))
                if dih_id[::-1] not in dihedrals:
                    dihedrals[dih_id] = dih_type
            #"right" side dihedrals k[2]
            for neib in neibs[k[2]]:
                if neib in k: continue
                dih_id,dih_type = Interactions.sorted_id_and_type(types,(k[0],k[1],k[2],neib))
                if dih_id[::-1] not in dihedrals:
                    dihedrals[dih_id] = dih_type
        tf = perf_counter()
        logger.debug('dihedrals time --> {:.3e} sec'.format(tf-t0))
        return dihedrals
    
    @staticmethod
    def sorted_id_and_type(types,a_id):
        t = [types[i] for i in a_id]
        if t[0] <= t[-1]:
            t = tuple(t)
        else:
            t = tuple(t[::-1])
        if a_id[0]<=a_id[-1]:
            a_id = tuple(a_id)
        else:
            a_id = tuple(a_id[::-1])
        return a_id,t
    
    @staticmethod
    def get_neibs(connectivity):
        '''
        Computes first (bonded) neihbours of a system in dictionary format
        key: atom_id
        value: set of neihbours
        '''
        neibs = dict()
        if type(connectivity) is dict:
            for k in connectivity.keys(): 
                neibs[k[0]] = set() # initializing set of neibs
                neibs[k[1]] = set()
            for j in connectivity.keys():
                neibs[j[0]].add(j[1])
                neibs[j[1]].add(j[0])
        else:
            for i in range(connectivity.shape[0]):
                k1 =connectivity[i,0]
                k2 = connectivity[i,1]
                neibs[k1] = set()
                neibs[k2] = set()
            for i in range(connectivity.shape[0]):
                k1 =connectivity[i,0]
                k2 = connectivity[i,1]
                neibs[k1].add(k2)
                neibs[k2].add(k1)
            
        return neibs
    
    @staticmethod
    def get_unconnected_structures(neibs):
        '''
        The name could be "find_unbonded_structures"
        Computes an array of sets. The sets contain the ids of a single molecule.
        Each entry corresponds to a molecule
        '''
        unconnected_structures=[]
        for k,v in neibs.items():
            #Check if that atom is already calculated
            in_unstr =False
            for us in unconnected_structures:
                if k in us:
                    in_unstr=True
            if in_unstr:
                continue
            sold = set()
            s = v.copy() #initialize the set
            while sold != s:
                sold = s.copy()
                for neib in sold:
                    s = s | neibs[neib]
            unconnected_structures.append(s)
        try:
            sort_arr = np.empty(len(unconnected_structures),dtype=int)
            for i,unst in enumerate(unconnected_structures):
                sort_arr[i] =min(unst)
            x = sort_arr.argsort()
            uncs_struct = np.array(unconnected_structures)[x]
        except ValueError as e:
            logger.error('{}'.format(e))
            uncs_struct = np.array(unconnected_structures)
            
        return uncs_struct
    
    @staticmethod
    def get_at_types(at_types,bonds):
        at_types = at_types.copy()
        n = len(at_types)
        db = np.zeros(n,dtype=int)
        for b in bonds:
            if b[2] ==2:
                db [b[0]] +=1 ; db [b[1]] +=1
        for i in range(n):
            if db[i] == 0 :
                pass
            elif db[i] ==1:
                at_types[i] +='='
            elif db[i]==2:
                at_types[i] = '='+at_types[i]+'='
            else:
                logger.debug('for i = {:d} db = {:d} . This is not supported'.format(i,db[i]))
                raise Exception(NotImplemented)
                
        return at_types
    
    @staticmethod
    def get_united_types(at_types,neibs):
        united_types = np.empty(len(at_types),dtype=object)
        for i,t in enumerate(at_types):
            print(i,t)
            nh=0
            if i in neibs:
                for neib in neibs[i]:
                    if at_types[neib].strip()=='H':nh+=1
                
            if  nh==0: united_types[i] = t
            elif nh==1: united_types[i] = t+'H'
            elif nh>1 : united_types[i] = t+'H'+str(nh)
        
        return united_types
    
    @staticmethod
    def get_itypes(model,at_types,bonds,neibs):
        logger.debug('atom model = {:s}'.format(model))
        if model.lower() in ['ua','united-atom','united_atom']:
            itypes = Interactions.get_united_types(at_types,neibs)
        elif model.lower() in ['aa','all-atom','all_atom']:
            itypes = Interactions.get_at_types(at_types, bonds)
        else:
            logger.debug('Model "{}" is not implemented'.format(model))
            raise Exception(Not)
        
        return itypes

    @staticmethod
    def get_vdw_unconnected(types,unconnected_structures):
        vdw_uncon = dict()
        for i1,unst1 in enumerate(unconnected_structures):
            for i2,unst2 in enumerate(unconnected_structures):
                if i2 <= i1:
                    continue
                logger.debug('Adding interactions between structs {},{}'.format(i1,i2))
                for j1 in list(unst1):
                    for j2 in list(unst2):
                        vid, ty = Interactions.sorted_id_and_type(types,(j1,j2) )
                        vdw_uncon[vid] = ty
        
        return vdw_uncon
    
    @staticmethod
    def get_vdw_connected(types,unconnected_structures,neibs,vdw_bond_dist):
        #Adding by structure
        #2.2) Calculate number of bonds between each pair of atoms 
        # March through the pairs and for each one,
        # add the neibours of the neibours of the first atom until you find the other
        # Bonds between atoms is equivalent to the times we look for new neibours. 
        #Distances between united atoms
        vdw_con = dict()
        for unst in unconnected_structures:
            uns = list(unst)
            set_uns = set(uns)
            for j in uns:
                s = neibs[j].copy()
                nbonds = 1
                neibs_of_j = []
                sets_uniqual =True 
                while sets_uniqual:        
                    current_neib = s.copy()
                    current_neib.discard(j)
                    for nj in neibs_of_j:
                        current_neib.discard(nj)
                    for jneib in current_neib:
                        #if jneib<j: continue
                        neibs_of_j.append(jneib)
                        if nbonds >= vdw_bond_dist:
                            vid,ty = Interactions.sorted_id_and_type(types, (j,jneib) )
                            vdw_con[vid] = ty  
                    sets_uniqual = s !=set_uns
                    for neib in s.copy(): s = s | neibs[neib]
                    nbonds+=1
        
        return vdw_con
    
    @staticmethod
    def get_vdw(types,neibs,find_vdw_connected,
                 find_vdw_unconnected,
                 vdw_bond_dist=4):
        if find_vdw_connected or find_vdw_unconnected:
            #Find unconnected structures
            uncon_struct = Interactions.get_unconnected_structures(neibs)       
            #############################
        if find_vdw_unconnected:
            vdw_unc = Interactions.get_vdw_unconnected(types,uncon_struct)
        else:
            vdw_unc = dict()
        if find_vdw_connected:
            vdw_conn = Interactions.get_vdw_connected(types,uncon_struct,neibs,
                                                      vdw_bond_dist)
        else:
            vdw_conn = dict()
                
        vdw = dict()
        for d in [vdw_conn,vdw_unc]:
            vdw.update(d)
            
        return vdw
    
    @staticmethod
    def inverse_dictToArraykeys(diction):
        arr =  list( np.unique( list(diction.values()) , axis =0) )
       
        inv = {tuple(k) : [] for k in arr }
        for k,val in diction.items():
            inv[val].append(k)
        return {k:np.array(v) for k,v in inv.items()}
  
    @staticmethod
    def clean_hydro_inters(inters):
        #Remove Hydrogen atoms if you have to
        def del_inters_with_H(it):
            dkeys =[]
            for k in it.keys():
                if 'H' in k: dkeys.append(k)
            for k in dkeys:
                del it[k]
            return it
        for k,vv in inters.copy().items():
            inters[k] = del_inters_with_H(vv)
        
        return inters
    
    def get_rhos(vdw):
        find_densities_with = np.unique([list(ty) for ty in vdw])
        rhos = dict()
        
        for ty in find_densities_with:
            for tyvdw,arr in vdw.items():
                if ty == tyvdw[0]:
                    ltpa = tuple([ arr[arr[:,1]==i] for i in np.unique(arr[:,1]) ])
                    rhos[(tyvdw[1],tyvdw[0])] = ltpa # central atom first
                if ty == tyvdw[1]:
                    ltpa = tuple([ arr[arr[:,0]==i] for i in np.unique(arr[:,0]) ])
                    rhos[tyvdw] = ltpa # central atom first
                    
        return rhos
    
    def find_configuration_inters(self,Bonds,atom_types):
        
        Bonds = Interactions.bonds_to_python(Bonds)
        neibs = Interactions.get_neibs(Bonds)
        #at_types = Interactions.get_at_types(atom_types.copy(),Bonds)
        at_types=atom_types
        connectivity = Interactions.get_connectivity(Bonds,at_types,self.excludedBondtypes)
        
        neibs = Interactions.get_neibs(connectivity)
        types = Interactions.get_itypes(self.atom_model,at_types,Bonds,neibs)
        types = Interactions.get_at_types(types.copy(),Bonds)
        
        #1.) Find 2 pair-non bonded interactions
        vdw = Interactions.get_vdw(types,neibs,self.find_vdw_connected,
                      self.find_vdw_unconnected,
                      self.vdw_bond_dist)      


        if self.find_angles or self.find_dihedrals:
            angles = Interactions.get_angles(connectivity,neibs,types)
            
            
            if find_dihedrals:
                dihedrals = Interactions.get_dihedrals(angles,neibs,types)
    
            else:
                dihedrals = dict()
        else:
            angles = dict()
            dihedrals= dict()
            

        inters = {k : Interactions.inverse_dictToArraykeys(d) 
                  for k,d in zip(['connectivity','angles','dihedrals','vdw'],
                              [connectivity, angles, dihedrals, vdw])
                 }
                                                   
        if self.find_densities:
            rhos = Interactions.get_rhos(inters['vdw'])
        else:
            rhos = dict()
        inters['rhos'] = rhos
        
        if self.atom_model.lower() in ['ua','united-atom','united_atom']:    
            inters = Interactions.clean_hydro_inters(inters)
            
        return inters
     
    def InteractionsForData(self):
        t0 = perf_counter()
        dataframe = self.data
        
        All_inters = np.empty(len(dataframe),dtype=object)
        
        first_index = dataframe.index[0]
        Bonds = dataframe['Bonds'][first_index].copy()
        atom_types = dataframe['atoms_type'][first_index].copy()
        inters = self.find_configuration_inters(Bonds,atom_types)
        
        for i,(j,data) in enumerate(dataframe.iterrows()):
            b_bool =dataframe.loc[j,'Bonds'] != Bonds
            t_bool = dataframe.loc[j,'atoms_type'] != atom_types
            if b_bool or t_bool:
                Bonds = data['Bonds'].copy()
                atom_types = data['atoms_type'].copy()
                inters = self.find_configuration_inters(Bonds,atom_types)
                logger.info('Encountered different Bonds or Atoms_type on j = {}\n Different bonds --> {}  Different types --> {}'.format(j,b_bool,t_bool))
            All_inters[i] = inters
            
        dataframe['interactions'] = All_inters
        return
    
    @staticmethod
    def plot_phi_rho(r0,rc):
        def get_rphi(r0,rc):
            c = Interactions.compute_coeff(r0, rc)
            r = np.arange(0,rc*1.04,0.001)
            phi = np.array([Interactions.phi_rho(x,c,r0,rc) for x in r])
            return r,phi
        size = 3.2
        fig = plt.figure(figsize=(size,size),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=3)
        plt.tick_params(direction='in', which='major',length=5)
        plt.ylabel(r'$\phi(r)$')
        plt.xlabel(r'$r$ $(\AA)$')
        plt.xticks([i for i in range(int(max(rc))+1)])
        colors = ['#e41a1c','#377eb8','#4daf4a']
        plt.title('Illustration of the activation function',fontsize=3.4*size)
        styles =['-','-.','--']
        if iterable(r0):
            for i,(rf0,rfc) in enumerate(zip(r0,rc)):
                r,phi = get_rphi(rf0,rfc)
                label = r'$r_c$={:1.1f} $\AA$'.format(rfc)
                plt.plot(r,phi,label=label,color=colors[i%3],lw=size/2.0,ls=styles[i%3])
            plt.legend(frameon=False,fontsize=2.5*size)
        else:
            r,phi = get_rphi(r0,rc)
            plt.plot(r,phi)
        plt.savefig('activation.eps',format='eps',bbox_inches='tight')
        plt.show()
    @staticmethod
    def compute_coeff(r0,rc):
        c = np.empty(4,dtype=float)
        r2 = (r0/rc)**2
        r3 = (1.0 -r2)**3
        c[0] = (1.0-3.0*r2)/r3
        c[1] =(6.0*r2)/r3/rc**2
        c[2] = -3.0*(1.0+r2)/r3/rc**4
        c[3] = 2.0/r3/rc**6
        return c
    
    @staticmethod
    def phi_rho(r,c,r0,rc):
        if r<=r0:
            return 1
        elif r>=rc:
            return 0
        else:    
            return c[0] + c[1]*r**2 + c[2]*r**4 + c[3]*r**6
    
    def calc_interaction_values(self):
        n = len(self.data)
        all_Values = np.empty(n ,dtype=object)
        atom_confs = np.array(self.data['atomic_configuration'])
        interactions = np.array(self.data['interactions'])
        for i,ac,inters in zip(range(n),atom_confs,interactions):
            values = dict(keys=inters.keys())
            for intertype,vals in inters.items():
                d = {t:[] for t in vals.keys()}
                for t,pairs in vals.items():
                    temp = np.empty(len(pairs),dtype=float)
                    if intertype in ['connectivity','vdw']:
                        for im,p in enumerate(pairs):
                            r1 = np.array(ac[p[0]]) ; r2 = np.array( ac[p[1]])
                            temp[im] =  calc_dist(r1,r2)
                    elif intertype=='angles':
                        for im,p in enumerate(pairs):
                            r1 = np.array(ac[p[0]]) ; 
                            r2 = np.array(ac[p[1]]) ; 
                            r3 = np.array(ac[p[2]])
                            temp[im] = calc_angle(r1,r2,r3)
                    elif intertype =='dihedrals':
                        for im,p in enumerate(pairs):
                            r1 = np.array(ac[p[0]]) ; 
                            r2 = np.array(ac[p[1]]) ; 
                            r3 = np.array(ac[p[2]])
                            r4 = np.array(ac[p[3]])
                            temp[im] = calc_dihedral(r1, r2, r3, r4)
                        
                    elif intertype=='rhos':
                        rhoi = dict()
                        for ldi,(r0,rc) in enumerate(zip(self.rho_r0,self.rho_rc)):
                            temp = np.empty(len(pairs),dtype=float)
                            c = self.compute_coeff(r0, rc)
                            for im,ltpa in enumerate(pairs):
                                rho = 0
                                for p in ltpa:
                                    r1 = np.array(ac[p[0]]) ; r2 = np.array( ac[p[1]])
                                    r12 = calc_dist(r1,r2)
                                    rho += self.phi_rho(r12,c,r0,rc)
                    
                                temp[im] = rho
                            rhoi[str(ldi)] = temp.copy()
                        temp = rhoi
                    else:
                        raise Exception(NotImplemented)
                    
                    d[t] = temp.copy()
                
                if intertype!='rhos':
                    values[intertype] = d.copy()
                else:
                    td = dict()
                    for t,v in d.items():
                        for ldi,rhoi in v.items():
                            k = intertype + ldi
                            if k not in td:
                                td[k]=dict()
                            td[k][t] = rhoi
                    
                    values.update(td)
            
            all_Values[i] = values
        self.data['values'] = all_Values
        return



class Data_Manager():
    def __init__(self,data,setup):
        self.data = data
        self.setup = setup
        return
    
    
    @staticmethod
    def data_filter(data,selector=dict(),operator='and'):
        
        n = len(data)
        if selector ==dict():
            return np.ones(n,dtype=bool)
        
        if operator =='and':
            filt = np.ones(n,dtype=bool)
            oper = np.logical_and
        elif operator =='or':
            filt = np.zeros(n,dtype=bool)
            oper = np.logical_or
            
        for k,val in selector.items():
            if iterable(val):
                f1 = np.zeros(n,dtype=bool)
                for v in val: f1 = np.logical_or(f1, data[k] == v)
            else:
                 f1 = data[k] == val
            filt = oper(filt,f1)
                
        return filt
    
    def select_data(self,selector=dict(),operator='and'):
        filt = self.data_filter(self.data,selector,operator)
        return self.data[filt]
    
    @staticmethod
    def generalized_data_filter(data,selector=dict()):
        n = len(data)
        filt = np.ones(n,dtype=bool)
        for k,val in selector.items():
            try:
                iter(val)
            except:
                s = 'val must be iterable, containing [operator or [operators], value or [values]]'
                logger.error(s)
                raise Exception(s)
            else:
                try:
                    iter(val[1])
                    iter(val[0])
                    f1 = np.zeros(n,dtype=bool)
                    for operator,v in zip(val[0],val[1]): 
                        f1 = np.logical_or(f1, operator(data[k],v))
                except:
                    f1 = val[0](data[k],val[1])
            filt = np.logical_and(filt,f1)
        return filt
    
    def clean_data(self,cleaner=dict()):
        # default are dumb values
        filt = self.generalized_data_filter(self.data,selector=cleaner)
        self.data = self.data[filt]
        return    
    
    @staticmethod
    def save_selected_data(fname,data,selector=dict()):
        
        dataT = data[Data_Manager.data_filter(data,selector)]
        with open(fname,'w') as f:
            for j,data in dataT.iterrows():
                at = data['atoms_type']
                ac = data['atomic_configuration']
                na = data['natoms']
                try:
                    comment = data['comment']
                except:
                    comment = ''
                f.write('{:d} \n{:s}\n'.format(na,comment))
                for j in range(na):
                    f.write('{:3s} \t {:8.6f} \t {:8.6f} \t {:8.6f}  \n'.format(at[j],ac[j][0],ac[j][1],ac[j][2]) )
            f.closed
            
        return 
    
    @staticmethod
    def read_frames(filename):
        with open(filename,'r') as f:
            lines = f.readlines()
            f.closed
     
        line_confs =[]
        natoms = []
        comments = []
        for i,line in enumerate(lines):
            if 'iteration' in line:
                line_confs.append(i-1)
                comments.append(line.split('\n')[0])
                natoms.append(int(lines[i-1].split('\n')[0]))
        
        confs_at = []
        confs_coords = []
        for j,n in zip(line_confs,natoms):
           
            atoms_type = []
            coords = []
            for i in range(j+2,j+n+2):         
                li = lines[i].split('\n')[0].split()
                atoms_type.append( li[0] )
                coords.append(np.array(li[1:4],dtype=float))
            confs_at.append(atoms_type)
            confs_coords.append(np.array(coords))
       
        def ret_val(a,c):
            for j in range(len(c)):            
                if a == c[j]:
                    return c[j+2]
            return None
        
        energy = [] ; constraint = [] ; cinfo = [] ;maxf=[] ; con_val =[] 
        for comment in comments:
            c = comment.split()
            cnm = ret_val('constraint',c)
            cval = ret_val('constraint_val',c)
            cinfo.append(cnm+cval[0:5])
            constraint.append(cnm)
            con_val.append(float(cval))
            energy.append(float(ret_val('eads',c)))
            maxf.append(float(ret_val('maxforce',c)))
            
        #cons = np.unique(np.array(constraint,dtype=object))
    
        data  = {'Energy':energy,'cinfo':cinfo,'constraint':constraint,'constraint_val':con_val,'max_force':maxf,'natoms':natoms,
                 'atoms_type':confs_at,'atomic_configuration':confs_coords,'comment':comments}
        dataframe =pd.DataFrame(data)
        #bonds
        

        return dataframe 
    
    @staticmethod
    def make_labels(dataframe):
        n = len(dataframe)
        attr_rep = np.empty(n,dtype=object)
        i=0
        for j,data in dataframe.iterrows():
            
            if 'rep' in data['cinfo']:
                attr_rep[i]='rep'
            else:
                attr_rep[i]='attr'
            i+=1 
            
        dataframe['label'] = 'inter'
        dataframe['attr_rep'] = attr_rep
        #labels
        for c in dataframe['cinfo'].unique():
            filt = dataframe['cinfo'] == c
            idxmin = dataframe['Energy'][filt].index[dataframe['Energy'][filt].argmin()]
            dataframe.loc[idxmin,'label'] = 'optimal'
        return
    

    
    def read_mol2(self,filename,read_ener=False,label=False):
        import numpy as np
        # first read all lines
        with open(filename, 'r') as f:	#open the text file for reading
            lines_list = f.readlines()
            f.closed
        logger.info("reading mol2 file: {}".format(filename))
        #Find starting point,configuration number, configuration distance
        nconfs = 0
        lns_mol = []; lns_atom = [] ; lns_bond = []
        for i,line in enumerate(lines_list):
            if line.split('\n')[0] == '@<TRIPOS>MOLECULE':
                 nconfs += 1
                 lns_mol.append(i)
            if line.split('\n')[0] == '@<TRIPOS>ATOM':
                 lns_atom.append(i)
            if line.split('\n')[0] == '@<TRIPOS>BOND':
                 lns_bond.append(i)
    
        logger.info(' Number of configurations = {:d}'.format(nconfs))
        if nconfs != len(lns_atom) or len(lns_bond) != len(lns_atom):
            logger.error('nconfs = {:d}, ncoords = {:d}, nbonds = {:d}'.format(nconfs,len(lns_atom),len(lns_bond)))
      
        natoms = 30000 ; nbonds = 50000; # default params
        Natoms = [] 
        Atom_Types =[] ; Bonds = [] ; Atom_coord = []
        if read_ener:
            energy = []
        if label:
            labels = []
        for iconf,(im,ia,ib) in enumerate(zip(lns_mol,lns_atom,lns_bond)):
            #reading natoms,nbonds
            natoms = int (lines_list[im+2].split('\n')[0].split()[0])
            nbonds = int (lines_list[im+2].split('\n')[0].split()[1])
            atoms_type = []
            bonds = []
            at_coord = []
            #Reading type of atoms and coordinates
            for ja in range(0,natoms):
                line = lines_list[ia+1+ja].split('\n')[0].split()
                atoms_type.append( line[5].split('.')[0] )
                at_coord.append([float(line[2]), float(line[3]),float(line[4])])
            # Reading bonds
            for jb in range(0,nbonds):
                line = lines_list[ib+1+jb].split('\n')[0].split()
                bonds.append(line[1:4])
            Atom_Types.append(atoms_type)
            Atom_coord.append(at_coord)
            Bonds.append(bonds)
            Natoms.append(natoms)
            if read_ener:
                energy.append(float(lines_list[im+1].split('\n')[0].split()[6]))
            #.debug('conf {:d}: natoms = {:d}, nbonds = {:d}'.format(iconf,natoms,nbonds))
    
        data_dict ={'atomic_configuration':Atom_coord,'atoms_type': Atom_Types,
                    'Bonds': Bonds,'natoms':Natoms}
        if read_ener:
            data_dict['Energy'] = energy
        return data_dict
    
    @staticmethod
    def read_Gaussian_output(filename,add_zeroPoint_corr=True,
                             units='kcal/mol',
                             clean_maxForce=None,
                             minEnergy=True,
                             enerTol=1e-6):
        import numpy as np
        import pandas as pd
        # first read all lines
        with open(filename, 'r') as f:	#open the text file for reading
            lines_list = f.readlines()
            f.closed
        logger.debug('reading filename "{:s}"'.format(filename))
        lc = [] ; 
        
        def key_in_line(key,lin):
            x = True
            if iterable(key):
                for ek in key:
                    if ek not in lin:
                        x=False
                        break
            else:
                x = key in lin
            return x
        
        leners = []
        forcelines = []
        for i,line in enumerate(lines_list):
            lin = line.split()
            
            if key_in_line(['Standard','orientation:'],lin): 
                lc.append(i+5)
            if key_in_line(['SCF','Done:','='],lin):
                if ' >>>>>>>>>> Convergence criterion not met.' in lines_list[i-1]:
                    approx_ener_line = i
                    continue                
                leners.append(i)
                
            if key_in_line(['Atomic','Forces','(Hartrees/Bohr)'],lin):
                forcelines.append(i+3)

        #Count number of atoms in file
        if len(leners) ==0:
            try:
                leners.append(approx_ener_line)
            except UnboundLocalError:
                pass
        natoms = 0
        atoms_type = []
        for line in lines_list[lc[0]::]:
            if line[5:7]=='--':
                break
            atoms_type.append(line.split()[1])
            natoms+=1
            
        
        '''
        striped_lines = [line.strip('\n ') for line in lines_list]
        eners = ''.join(striped_lines).split('HF=')[-1].split('\\')[0].split(',')
        eners = np.array(eners,dtype=float)
        '''
        striped_lines = [line.strip('\n ') for line in lines_list]
        eners = np.array([float(lines_list[i].split('=')[-1].split()[0]) 
                              for i in leners])
        if add_zeroPoint_corr:
            st = 'ZeroPoint='
            if st in ''.join(striped_lines):
                zero_point_corr = ''.join(striped_lines).split(st)[-1].split('\\')[0].split(',')
                eners-= np.array(zero_point_corr,dtype=float)
        
        
        
        for line in striped_lines:
            if 'Stoichiometry' in line:
                sysname = line.split()[-1]
        
        #drop last line
        logger.debug('found eners = {:d} ,found confs = {:d}'.format(eners.shape[0],len(lc)))

       
        if eners.shape[0] +1 == len(lc):
            lc = lc[:-1]
        
        if units=='kcal/mol':
            eners *= 627.5096080305927 # to kcal/mol
        Atypes = mappers().atomic_num
        for j in range(len(atoms_type)):
            atoms_type[j] = Atypes[atoms_type[j]]
        #Number of configurations
        nconfs = len(lc)
        logger.info('Reading file: {:} --> nconfs = {:d} ,ener values = {:d},\
                    natoms = {:d}'.format(filename,nconfs,len(eners),natoms))
        
        #Fill atomic configuration 
        config = []
        for j,i in enumerate(lc):
            coords = []
            for k,line in enumerate(lines_list[i:i+natoms]):
                li = line.split()
                coords.append(np.array(li[3:6],dtype=float))
            config.append( np.array(coords) )
        if clean_maxForce is not None:
        #if clean_maxForce is not None and eners.shape[0]>1:
            forces_max = []
            for j,i in enumerate(forcelines):
                forces= []
                for k,line in enumerate(lines_list[i:i+natoms]):
                    li = line.split()
                    forces.append(np.array(li[3:6],dtype=float))
                m = np.abs(forces).max()
                if units =='kcal/mol': m*=627.5096080305927
                forces_max.append(m)
            forceFilt = np.array(forces_max) < clean_maxForce
            
            logger.debug('ForceFilt shape = {}'.format(forceFilt.shape))
            config = [config[i] for i,b in enumerate(forceFilt) if b]
            eners = eners[forceFilt]
            
            nconfs = len(config)
            logger.info('NEW AFTER CLEANING nconfs = {:d} ,ener values = {:d},\
                    natoms = {:d}'.format(nconfs,len(eners),natoms))
        if nconfs ==0:
            
            raise ValueError
        
        if minEnergy:
            me = eners.argmin()
            config = [config[me]] # list
            eners = eners[[me]] #numpy array
            nconfs = eners.shape[0]
            
        data_dict ={'atomic_configuration':config,'natoms':[int(natoms)]*nconfs,
                    'atoms_type':[atoms_type]*nconfs,'sys_name':[sysname]*nconfs,
                    'filename':[filename]*nconfs}
        data_dict['Energy'] = eners
        #print(data_dict)
        data = pd.DataFrame(data_dict)
        
        return data
    

    def assign_system(self,sys_name):
        self.data['sys_name'] = sys_name
        return
    
    def create_dist_matrix(self):
        logger.info('Calculating distance matrix for all configurations \n This is a heavy calculation consider pickling your data')
        i=0
        dataframe = self.data
        size = len(dataframe)
        All_dist_m = np.empty(size,dtype=object)
        at_conf = dataframe['atomic_configuration'].to_numpy()
        nas = dataframe['natoms'].to_numpy()
        for i in range(size):
            natoms = nas[i]
            conf = at_conf[i]
            dist_matrix = np.empty((natoms,natoms))
            for m in range(0,natoms):
                for n in range(m,natoms):
                    r1 = np.array(conf[n])
                    r2 = np.array(conf[m])
                    rr = norm2(r2-r1)
                    dist_matrix[m,n] = rr
                    dist_matrix[n,m] = rr
            All_dist_m[i] = dist_matrix
            if i%1000 == 0:
                logger.info('{}/{} completed'.format(i,size))
        
        dataframe['dist_matrix'] = All_dist_m
        return  
    
    
    def sample_randomly(self,perc,data=None,seed=None):
        if data is None:
            data = self.data
        if not perc>0 and not perc <1:
            raise Exception('give values between (0,1)')  
        size = int(len(data)*perc)
        if seed  is not None: np.random.seed(seed)
        indexes = np.random.choice(data.index,size=size,replace=False)
        return data.loc[indexes]
    
    def train_test_valid_split(self):
        
        data = self.data
        train_perc = self.setup.train_perc
        seed = self.setup.seed
        sampling_method = self.setup.sampling_method
        
        if sampling_method=='random':
            train_data = self.sample_randomly(train_perc, data, seed)
            train_indexes = train_data.index
            
            valid_test_data = data.loc[data.index.difference(train_indexes)]
            
            ftest = np.random.choice([True,False],len(valid_test_data))
            
            test_data = valid_test_data[ftest]
            
            valid_data = valid_test_data[np.logical_not(ftest)]
        
        elif sampling_method=='column':
            
            vinfo = self.setup.validation_set
            
            ndata = len(data)
            f = np.ones(ndata,dtype=bool)
            
            for colval in vinfo.split('&'):
                temp =  colval.split(':')
                col = temp[0]
                vals = temp[1]
                fv = np.zeros(ndata,dtype=bool)
                for v in vals.split(','):
                    fv = np.logical_or(fv,data[col]==v.strip())
                f = np.logical_and(fv,f)
            
            valid_data = data[f]
            train_test_data = data[np.logical_not(f)]
            train_data = self.sample_randomly(train_perc, train_test_data, seed)
            train_indexes = train_data.index
            test_data = train_test_data.loc[ train_test_data.index.difference(train_indexes) ]
            
        else:
            raise Exception(NotImplementedError('"{}" is not a choice'.format(sampling_method) ))
        test_indexes = test_data.index
        valid_indexes = valid_data.index
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.valid_indexes = valid_indexes
        return train_indexes,test_indexes,valid_indexes
    
    def bootstrap_samples(self,nsamples,perc=0.3,seed=None,
                          sampling_method='random',nbins=100,bin_pop=1):
        if seed is not None:
            np.random.seed(seed)
        seeds = np.random.randint(10**6,size = nsamples)
        boot_data = []
        for seed in seeds: 
            if sampling_method =='random':
               boot_data.append(self.sample_randomly(perc,seed))
            elif sampling_method =='uniform_energy':
                boot_data.append(self.sample_energy_data_uniformly(nbins,bin_pop))
        return boot_data
    
    def sample_energy_data_uniformly(self,nbins=100,bin_pop=1):
        E = np.array(self.data['Energy'])
        Emin = E.min()
        Emax = E.max()
        dE = (Emax - Emin)/float(n_bins)
        indexes = []
      #  ener_mean_bin = np.empty(n_bins,dtype=float)
     #   ncons_in_bins = np.empty(n_bins,dtype=int)
        for i in range(0,n_bins-1):
            # Find confs within the bin
            ener_down = Emin+dE*float(i)#-1.e-15
            ener_up   = Emin+dE*float(i+1)#+1.e-15
            fe = np.logical_and(ener_down<=E,E<ener_up )
            temp_conf = np.array([j for j in self.data[fe].index])
            if temp_conf.shape[0] >= bin_pop: 
                con = np.random.choice(temp_conf,size=bin_pop)
            else:
                con = temp_conf
            indexes += con 
        return self.data.loc[indexes]
    
    def setup_bonds(self,distance_setup):
        self.create_dist_matrix()
        data = self.data
        distance_setup = {tuple(np.sort(k)):v for k,v in distance_setup.items()}
        size = len(data)
        bonds = np.empty(size,dtype=object)

        index = data.index
        for i in range(size):
            j = index[i]
            
            natoms = data.loc[j,'natoms']
            
            dists = data.loc[j,'dist_matrix']
            at_types = data.loc[j,'atoms_type']
            bj = []
            for m in range(0,natoms):
                for n in range(m+1,natoms):
                    ty = tuple(np.sort([at_types[m],at_types[n]]))
                    if ty not in distance_setup:
                        continue
                    else:
                        d = dists[m,n]
                        if d <= distance_setup[ty][0]:
                            bj.append([m,n,2])
                        elif d<= distance_setup[ty][1]:
                            bj.append([m,n,1]) 
            bonds[i] = bj
            
                    
        data['Bonds'] = bonds
        return
    
    @staticmethod
    def get_pair_distance_from_data(data,atom1,atom2):
        confs = data['atomic_configuration'].to_numpy()
        dist = np.empty(len(data),dtype=float)
        for i,conf in enumerate(confs):
            r1 = conf[atom1]
            r2 = conf[atom2]
            dist[i] = norm2(r2-r1)
        return dist
            
    @staticmethod
    def get_pair_distance_from_distMatrix(data,atom1,atom2):
        distM = data['dist_matrix'].to_numpy()
        dist = np.empty(len(data),dtype=float)
        for i,cd in enumerate(distM):
            dist[i] = cd[atom1,atom2]
        return dist
    
    def get_systems_data(self,data,sys_name=None):
        if sys_name == None:
            if len(np.unique(data['sys_name']))>1:
                raise Exception('Give sys_name for data structures with more than one system')
            else:
                data = self.data
        else:
            data = self.data [ self.data['sys_name'] == sys_name ] 
            
        return data
    def plot_distribution(self,ty,inter_type='vdw',bins=100,ret_max=False):
        dists = self.get_distribution(ty,inter_type)
        fig = plt.figure(figsize=(3.5,3.5),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=3)
        plt.tick_params(direction='in', which='major',length=5)
        plt.ylabel('distribution')
        if inter_type in ['vdw','connectivity']:
            plt.xlabel('r({}) \ $\AA$'.format('-'.join(ty)))
        else:
            plt.xlabel(r'{:s}({:s}) \ '.format(inter_type,'-'.join(ty)))
        plt.hist(dists,bins=bins,histtype='step',color='magenta')
        plt.show()
        if ret_max:
            return dists.max()
        return
    def get_distribution(self,ty,inter_type='vdw'):
        data = self.data
        #data = self.get_systems_data(self.data,sys_name)
        pair_dists = []
        #key = tuple(np.sort([t for t in ty]))
        for j,d in data.iterrows():
            pair_dists.extend(d['values'][inter_type][ty])
        return np.array(pair_dists)
    

    
class Optimizer():
    def __init__(self,data, train_indexes,test_indexes,valid_indexes,setup):
        if isinstance(data,pd.DataFrame):
            self.data = data
        else:
            raise Exception('data are not pandas dataframe')
        '''
        for t in [train_indexes,test_indexes,valid_indexes]:
            if not isinstance(t,type(data.index)):
                s = '{} is not pandas dataframe index'.format(t)
                logger.error(s)
                raise Exception(s)
        '''
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.valid_indexes = valid_indexes
        if isinstance(setup,Setup_Interfacial_Optimization):
            self.setup = setup
        else:
            s = 'setup is not an appropriate object'
            logger.error(s)
            raise Exception(s)
        return
    
    @property
    def data_train(self):
        return self.data.loc[self.train_indexes]
    @property
    def data_test(self):
        return self.data.loc[self.test_indexes]
    @property
    def data_valid(self):
        return self.data.loc[self.valid_indexes]

        
    
class Interfacial_FF_Optimizer(Optimizer):
    def __init__(self,data,train_indexes,test_indexes,valid_indexes,setup):
        super().__init__(data,train_indexes,test_indexes,valid_indexes,setup)
        
        return
    def get_indexes(self,dataset):
        if dataset=='train':
            return self.train_indexes
        elif dataset=='test':
            return self.test_indexes
        elif dataset =='valid':
            return self.valid_indexes
        elif dataset =='all':
            return self.data.index
        else:
            raise Exception('available options are {"train", "test", "valid", "all"}')
    
    
    def get_nessesary_info(self,which,dataset):
        infoObjects = []
        npars = []
        bounds =  []
        regArgs = np.empty(0,dtype=int)
        params = np.empty(0,dtype=float)
        #prepare and serialize the distances
        for prefix in ['PW','LD']:
            npots = getattr(self.setup,'n'+prefix)
            for i in range(npots):
                initpars = getattr(self.setup,'{:s}{:s}{:d}'.format(which,prefix,i))
                pairvalues =self.get_pairvalues(prefix, i)
                prms,bnds,minfo = self.prepare_data(initpars,prefix,pairvalues,
                                                dataset=dataset)
                new_reg_args = minfo.regargs +sum(npars)
                
                regArgs = np.concatenate( ( regArgs,  new_reg_args) ) 
                params = np.concatenate((params,prms))
                print('{:s}{:s}{:d}'.format(which,prefix,i),new_reg_args,params.shape[0])
                bounds.extend(bnds)
                npars.append(prms.shape[0])
                
                infoObjects.append(minfo)
        return params, bounds, regArgs,infoObjects,np.array(npars,dtype=int)
    
    def Upair_ondata(self,which='opt',dataset='all'):
        ndata = len(self.get_Energy(dataset))
        
        params,bounds,regArgs,infoObjects,npars = self.get_nessesary_info(which,dataset)
            
        Uclass = self.computeUclass(params, ndata, infoObjects)
        index = self.get_indexes(dataset)
        self.data.loc[index,'Uclass'] = Uclass
        return Uclass

    @staticmethod
    @jit(nopython=True,parallel=True)
    def Upair(Uclass,npars_pertype,Np,u_model,dists,dl,du,model_pars):
        #compute upair
        dix = 0
        for t in range(Np.size):
            n = Np[t]# number of distances per type
            d = dists[dix:dix+n]
            #print(d.shape)
            m = model_pars[t*npars_pertype:(t+1)*npars_pertype]
            #print(m.shape)
            U = u_model(d,m)
            dix+=n
            dlt = dl[t]  # shape numper of data
            dut = du[t]  # shape number of data
            
            for i in prange(Uclass.size):
                Uclass[i] += U[dlt[i] : dut[i]].sum()
        
        return 
   
    
    @staticmethod
    def computeUclass(params,ne,infoObjects):
        Uclass = np.zeros(ne,dtype=float)
        npars_old = 0
        for minf in infoObjects:
            #t0 = perf_counter()
            npars_new = npars_old  + np.count_nonzero(minf.isnot_fixed)
            
            objparams = params[ npars_old : npars_new ]
            
            model_pars = Interfacial_FF_Optimizer.serialize_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,minf.Nt,
                                                    minf.npars_pertype
                                                    )
            #print('overhead {:4.6f} sec'.format(perf_counter()-t0))
            #compute Uclass
            Utemp = np.zeros(ne,dtype=float)
            Interfacial_FF_Optimizer.Upair(
                                Utemp,minf.npars_pertype,minf.Np,minf.u_model,
                                minf.dists,minf.dl,minf.du,model_pars
                                )
            Uclass+=Utemp
            #print('computation {:4.6f} sec'.format(perf_counter()-t0))
            npars_old = npars_new
        return Uclass
    
    @staticmethod
    def LossFunction(params,costf,Energy,we,infoObjects,reg,regArgs,regf):   
        # serialize the params
        ne = Energy.shape[0]
        Uclass = Interfacial_FF_Optimizer.computeUclass(params,ne,infoObjects)
        # Compute cos
        preg = params[regArgs]
        cost = costf(we*Energy,we*Uclass) + reg*regf(preg)
        return cost
    
    def find_model(self,model):
        try:
            u_model = globals()['u_'+model]
        except:
            raise Exception('model {} --> {}'.format(model,NotImplemented))
        return u_model
    
   
    @staticmethod
    def serialize_model_parameters(params,fixed_params,isnot_fixed,
                                   Nt,npars_pertype):
        model_pars = []
        tint = np.arange(0,Nt,1,dtype=int)
        k1 = 0 ; k2 = 0
        for t in tint:
            kt = t*npars_pertype
            for mj in range(npars_pertype):
                if isnot_fixed[kt+mj]: model_pars.append(params[k1]) ;k1 +=1
                else: model_pars.append(fixed_params[k2]) ; k2+=1
        model_pars = np.array(model_pars)
        return model_pars
    
    def serialize_values(self,data_dict,vdw_params,col):
        
        Nt = len(vdw_params)
        Np = np.zeros(Nt,dtype=int)
        ndata = len(data_dict)
        dl = np.empty((Nt,ndata),dtype=int)
        du = np.empty((Nt,ndata),dtype=int)
        
        dists = [] 
        t0 = perf_counter()

        for i1,t in enumerate(vdw_params.index):
            ndists_t =0
            for i2,(j,val) in enumerate(data_dict.items()):
                inters = val[col]             
                if t in inters.keys(): 
                    nd = inters[t].shape[0]
                    Np[i1] += nd
                    dists.extend(inters[t]) #extent works faster than numpy concatenation
                else:
                    nd = 0
                dl[i1,i2] = ndists_t
                du[i1,i2] = ndists_t + nd
                
                ndists_t += nd
                
        dists = np.array(dists)
        if np.sum(Np) != dists.shape[0]:
            s = 'Something is wrong with the serialization sum Np {:3d} != dists.shape[0] {:3d}'.format(np.sum(Np),dists.shape[0])
            logger.error(s)
            raise Exception(s)
        
        return dists,dl,du,Np,Nt
    
    def get_Energy(self,dataset):
        if dataset =='train':
            E =  np.array(self.data_train['Energy'])
        elif dataset =='test':
            E =  np.array(self.data_test['Energy'])
        elif dataset =='valid':
            E =  np.array(self.data_valid['Energy'])
        elif dataset == 'all':
            E =  np.array(self.data['Energy'])
        else:
            s = "'dataset' only takes the values ['train','test','valid','all']"
            logger.error(s)
            raise Exception(s)
        return E
    
    def get_dataDict(self,dataset):
        if dataset =='train':
            data_dict = self.data_train['values'].to_dict()
        elif dataset =='test':
            data_dict = self.data_test['values'].to_dict()
        elif dataset =='valid':
            data_dict = self.data_valid['values'].to_dict()
        elif dataset == 'all':
            data_dict = self.data['values'].to_dict()
        else:
            s = "'dataset' only takes the values ['train','test','valid','all']"
            logger.error(s)
            raise Exception(s)
        return data_dict
    
    
    class Model_Info():
        def __init__(self,u_model,fixed_params,
                     isnot_fixed,npars_pertype,Nt,Np,dists,dl,du,
                     *args,**kwargs):
            self.u_model = u_model
            self.fixed_params = fixed_params
            self.isnot_fixed = isnot_fixed
            self.npars_pertype = npars_pertype
            self.Nt = Nt # number of types
            self.Np = Np # number of pairs per configuration and type
            self.dists = dists
            self.dl = dl
            self.du = du
            self.args = args
            for k,v in kwargs.items():
                setattr(self,k,v)
            return
        def __repr__(self):
            x = 'Attribute : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {} \n'.format(k,v)
            x+='--------------------\n'
            return x 
    
    def get_parameter_info(self,df_params,pairvalues):
        params = []; fixed_params = []; isnot_fixed = []; bounds =[]
        if pairvalues =='vdw':
            model_par_names = self.setup.model_parameters(self.setup.PW_model,
                                                    df_params.columns)
        elif pairvalues[:4]=='rhos':
            model_par_names = self.setup.model_parameters(self.setup.LD_model,
                                                    df_params.columns)
        else:
            raise NotImplemented
        for k,par in df_params.iterrows():
            for mn in model_par_names:
                isnot_fixed.append(par[mn+'_opt']) # its a question variable
                if par[mn+'_opt']:
                    params.append(par[mn])
                    bounds.append(tuple(par[mn+'_b']))
                    #print('TYPE = {} , par = {} , value = {:4.5f}'.format(k,mn,par[mn]))
                else: 
                    fixed_params.append(par[mn])
                
        params = np.array(params)
        fixed_params = np.array(fixed_params)
        isnot_fixed = np.array(isnot_fixed)
        npars_pertype = len(model_par_names)
        
        return params, bounds, fixed_params, isnot_fixed, npars_pertype,model_par_names
    
    @staticmethod
    def find_regularization_args(model,isnot_fixed):
        if 'Morse' in model:
            fu = lambda j : (j+2)%3==0
        elif 'double_exp' == model:
            fu = lambda j : j%3 == 0
        else:
            fu = lambda j : True
        regargs = []
        i=0
        for j,isf in enumerate(isnot_fixed): 
            if isf and fu(j):
                regargs.append(i)
            if isf:
                i+=1
        return np.array(regargs,dtype=int)
    
    def prepare_data(self,df_params,prefix,pairvalues,dataset = 'train'):
        # serialize the parameters
        t0 = perf_counter()
        model = getattr(self.setup, prefix+'_model')
        u_model = self.find_model(model)
        
        params, bounds, fixed_params, isnot_fixed, npars_pertype,model_par_names = self.get_parameter_info(df_params,pairvalues)        
        
        # choose the data
        data_dict = self.get_dataDict(dataset)
        
        # serialize distances and the amount of pairs for each type and data-point
        dists,dl,du, Np, Nt = self.serialize_values(data_dict,df_params,pairvalues)
        tf = perf_counter()
        logger.info('serialization time {:.3e} sec for {:4d} data'.format(tf - t0,len(data_dict)))
        
        regargs = self.find_regularization_args(model, isnot_fixed)
        
        #write cost function arguments

        minfo = self.Model_Info(u_model,fixed_params,isnot_fixed,npars_pertype,Nt,Np,
                               dists,dl,du,model_par_names=model_par_names,regargs=regargs)
        #print(minfo)
        return params,bounds,minfo
    
    @staticmethod
    def get_pairvalues(prefix,i):
        if 'PW'==prefix:
            pw = 'vdw'
        elif prefix =='LD':
            pw ='rhos'+str(i)
        else:
            raise NotImplemented
        return pw
    
    def optimize_params(self):
        t0 = perf_counter()
        
        E = self.get_Energy('train')
        weights = weighting(E,self.setup.weighting_method,self.setup.bT,self.setup.w)
        self.weights = weights
        
        params,bounds,regArgs,infoObjects,npars = self.get_nessesary_info( 'init','train')
        
        self.regArgs = regArgs

        opt_method = self.setup.optimization_method
        tol = self.setup.tolerance
        maxiter = self.setup.maxiter
        
        costf = getattr(measures(),self.setup.costf)
        
        LossFunc = self.LossFunction
        
        regf = getattr(regularizators(),self.setup.regularization_method)
        args = (costf,E,weights,infoObjects, self.setup.reg_par,regArgs,regf)
       
        self.infoObjects = infoObjects
        
        if params.shape[0] >0:
            t1 = perf_counter()
            
            if opt_method in ['SLSQP','BFGS','L-BFGS-B']:   
                logger.debug('I am performing {}'.format(opt_method))
                res = minimize(LossFunc, params,
                               args =args,
                               bounds=bounds,tol=tol, 
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':maxiter,
                                        'ftol': self.setup.tolerance},
                               method = opt_method)
                    
            elif opt_method =='DE':
                pops = int(self.setup.popsize)
                polish = bool( self.setup.polish)
                workers = 1 # currently only working with 1
                res = differential_evolution(LossFunc, bounds,
                            args =args,
                            maxiter = maxiter,
                            polish = self.setup.polish,
                             workers=workers ,
                            recombination=self.setup.recombination,
                            mutation = self.setup.mutation,
                            disp =self.setup.opt_disp)
            
            elif  opt_method == 'DA':
                #logger.debug('I am performing dual annealing')
                res = dual_annealing(LossFunc, bounds,
                            args =args ,x0=params,
                             maxiter = maxiter,
                             initial_temp=self.setup.initial_temp,
                             restart_temp_ratio=self.setup.restart_temp_ratio,
                             visit=self.setup.visit,
                             accept = self.setup.accept,
                             local_search_scale=self.setup.local_search_scale,
                             minimizer_kwargs={'method':'SLSQP',#
                                               'bounds':bounds,
                                               'options':{'maxiter':self.setup.maxiter_i,
                                                          'disp':self.setup.opt_disp,
                                                          'ftol':self.setup.tolerance},
                                },
                             disp=self.setup.opt_disp)
            elif opt_method =='direct':
                res = direct(LossFunc,bounds,args=args,locally_biased=False,maxiter=maxiter)
            else:
                s = 'Error: wrong optimization_method name'
                logger.error(s)
                raise Exception(s)
                
            self.opt_time = perf_counter() - t1
            optimization_performed=True
        else:
            optimization_performed = False
        
        if optimization_performed:
            self.regLoss = self.setup.reg_par*regf(res.x[regArgs])
            self.unregLoss = res.fun - self.regLoss  
            
            self.timef = 'time/fev = {:.3e} sec'.format(self.opt_time/res.nfev)
            logger.info(self.timef)
            
            logger.info('Total optimization time = {:.3e} sec'.format(self.opt_time))
            #Get the new params and give to vdw_dataframe
            self.results = res
             
        else:
            # Filling with dump values to avoid coding errors
            dump_value = 123321
            self.regLoss = dump_value ; self.unregLoss = dump_value ; self.results = dump_value
        self.optParams = dict()
        jiter =0
        for prefix in ['PW','LD']:
            n = getattr(self.setup,'n'+prefix)
            for i in range(n):
                init_dfX = getattr(self.setup,'init{:s}{:d}'.format(prefix,i))
                attrname= 'opt{:s}{:d}'.format(prefix,i)
                dicname = attrname[3:]
                if optimization_performed:
                    ni = np.sum(npars[:jiter])
                    ni1 = np.sum(npars[:jiter+1])
                    opt_X = res.x[ni : ni1]
                    optPars = self.save_params_todf( opt_X,init_dfX)
                    
                    
                    setattr(self,attrname,optPars)
                    setattr(self.setup,attrname,optPars)
                    self.optParams[dicname] = optPars
                    jiter+=1
                else:
                    setattr(self,attrname,init_dfX)
                    setattr(self.setup,attrname,init_dfX)
                    self.optParams[dicname] = init_dfX
                    
            
        return 
    
    def save_params_todf(self,arr_params,initdf):
        k=0
        df_params = initdf.copy()
        model_pars = [c for c in df_params.columns if '_' not in c]
        for t,par in df_params.iterrows():  
            for c in model_pars: 
                if par[c+'_opt']:
                    df_params.loc[[t],c] = arr_params[k] ; k+=1          
        return df_params
    
    def save_PW_potentials(self,typemap,pars=None):
        lines = []
        for k, params in self.optParams.items():
            if 'PW' not in k:
                continue
            if pars is None:
                pars = self.setup.model_parameters(self.setup.PW_model,params.columns)
            for ty,pot in params.iterrows():
                ty1 = typemap[ty[0]]
                ty2 = typemap[ty[1]]
                if ty2> ty1:
                    t = [ty1,ty2]
                else:
                    t=[ty2,ty1]
                parv = pot[pars]
                if self.setup.PW_model =='Morse':
                    add = 'morse/smooth/linear'
                else:
                    add = 'modify for not Morse'
                s = 'pair_coeff {:d} {:d} {:s} '.format(t[0],t[1],add)
                for p in parv:
                    s+=' {:6.4f} '.format(p)
                s+='\n'
                lines.append(s)
        with open(self.setup.runpath+'/PW.inc','w') as f:
            f.write(''.join(lines))
        return
    
    def save_LD_potentials(self,typemap,N_rho):
        N_LD = 0 
        data = self.data.loc[self.train_indexes]
        
        
        nblocks =0
        lc1 = '     # lower and upper cutoffs, single space separated'
        lc2 = '     # central atom types, single space separated'
        lc3 = '     # neighbor-types (neighbor atom types single space separated)'
        lc4 = '     # min, max and diff. in tabulated rho values, single space separated'
        
        lines = []
        for k,params in self.optParams.items():
            if 'LD' not in k:
                continue
            ni = int(k[2:])
            pars = self.setup.model_parameters(self.setup.LD_model,params.columns)
            fmodel = globals()['u_'+self.setup.LD_model] 
            num_pars = len(pars)
            for ty,pot in params.iterrows():
                x = np.array([pot[p] for p in pars])
                if (x == 0.0).all():
                    continue
                N_LD+=1
                r0 = self.setup.rho_r0[ni]
                rc = self.setup.rho_rc[ni]
                lines.append('{:.4e} {:.4e}  {:s}\n'.format(r0,rc,lc1))
                
                tyc = typemap[ty[1]]
                tynei = typemap[ty[0]]
                lines.append('{:d}  {:s}\n'.format(tyc,lc2))
                lines.append('{:d}  {:s}\n'.format(tynei,lc3))
                print('saving LD type {} on the same file'.format(ty))
                rho_distribution = Data_Manager(data,self.setup).get_distribution(ty,'rhos{:d}'.format(ni))
                rho_min = rho_distribution.min()
                rho_max = rho_distribution.max()
                diff = (rho_max-rho_min)/N_rho
                lines.append('{:.8e} {:.8e} {:.8e} {:s}\n'.format(rho_min,rho_max+diff,diff,lc4))
                
                rhov = np.arange(rho_min,rho_max+diff,diff)
                
                Frho = fmodel(rhov,x)
                for fr in Frho:
                    lines.append('{:.8e}\n'.format(fr))
                lines.append('\n')
            
        with open(self.setup.runpath +'/frhos{:d}.ld'.format(self.setup.nLD),'w') as f:
            f.write('{0:s}\n{0:s}Written by Force-Field Develop{0:s}\n{1:d} {2:d} # of LD potentials and # of tabulated values, single space separated\n\n'.format('******',N_LD,N_rho))
            for line in lines:
                f.write(line)
            f.closed
            
                
                
class regularizators:
    @staticmethod
    def ridge(x):
        return np.sum(x*x)
    @staticmethod
    def lasso(x):
        return np.sum(abs(x))
    @staticmethod
    def elasticnet(x):
        return 0.5*(regularizators.ridge(x)+regularizators.lasso(x))
    @staticmethod
    def smoothness(x):
        n2 = int(x.size/2)
        x1 = x[0:n2]
        x2 = x[n2:]
        sm = np.sum(np.abs(x1[2:] -2*x1[1:-1]+x1[0:-2])) + np.sum(np.abs(x2[2:] -2*x2[1:-1]+x2[0:-2]))
        return sm 
    @staticmethod
    def none(x):
        return 0.0
    
class measures:
    
    @staticmethod
    def M4(u1,u2):
        u = (u1-u2)**4
        return np.sum(u)/u1.shape[0]
    @staticmethod
    def relMAE(u1,u2):
        s = np.abs(u1).max()
        mae = measures.MAE(u1,u2)
        err = mae/s
        return 100*err
    @staticmethod
    def percE(u1,u2):
        s = np.abs(u1).max()
        mae = measures.MAE(u1,u2)
        mse = measures.MSE(u1,u2)
        return 0.5*(mae/s+mse/s**2)
    @staticmethod
    def elasticnet(u1,u2):
        return 0.5*(measures.MAE(u1,u2)+measures.MSE(u1,u2))
    @staticmethod
    def MAEmMSE(u1,u2):
        return np.sqrt(measures.MAE(u1,u2)*measures.MSE(u1,u2))
    @staticmethod
    def MAE(u1,u2):
        return np.abs(u1-u2).sum()/u1.shape[0]
    @staticmethod
    def MAEmMAX(u1,u2):
        u=np.abs(u1-u2)
        return u.sum()*u.max()/u1.shape[0]
    @staticmethod
    def STD(u1,u2):
        r = (u1 -u2)
        return r.std()
    @staticmethod
    def relMSE(u1,u2):
        mse = measures.MSE(u1,u2)
        s = np.abs(u1).max()
        
        e = 100*(mse**0.5)/s
        return e
    @staticmethod
    def MSE(u1,u2):
        u = u2 -u1
        return np.sum(u*u)/u1.shape[0]
    @staticmethod
    def L1(u1,u2):
        a = np.abs(u1)
        s = a.max()
        u = np.abs(u1-u2)/(s+a)
        return np.sum(u)/u1.shape[0]
    @staticmethod
    def L2(u1,u2):
        a = np.abs(u1)
        s = a.max()
        u = (u1-u2)**2/(s+a)
        return np.sum(u)/u1.shape[0]
    @staticmethod
    def BIAS(u1,u2):
        u = u2-u1
        return u.mean()
    @staticmethod
    def relBIAS(u1,u2):
        s = np.abs(u1.min())
        u= u2-u1
        return 100*u.mean()/s
    @staticmethod
    def blockMAE(u1,u2):
        nblocks=10
        u1min = u1.min()
        ran = u1.max()-u1min
        de = ran/nblocks
        bins = [(u1min+i*de,u1min+(i+1)*de) for i in range(nblocks) ]
        ut = 0
        fblocks=0
        for i in range(len(bins)):
            b =bins[i]
            
            f = np.logical_and(b[0]<=u1,u1<b[1])
            if not f.any():
                continue
            fblocks+=1
            u = np.abs(u1[f]-u2[f])
            ut+=u.mean()
        if fblocks == 0:
            return np.abs(u1-u2).mean()
        return ut/fblocks
    
    @staticmethod
    def BlockBIAS(u1,u2):
        nblocks=10
        u1min = u1.min()
        ran = u1.max()-u1min
        de = ran/nblocks
        bins = [(u1min+i*de,u1min+(i+1)*de) for i in range(nblocks) ]
        ut = 0
        fblocks=0
        for i in range(len(bins)):
            b =bins[i]
            f = np.logical_and(b[0]<=u1,u1<b[1])
            if not f.any():
                continue
            fblocks+=1
            u = u1[f]-u2[f]
            ut+=u.mean()
        if fblocks == 0:
            return (u1-u2).mean()
        return ut/fblocks

    @staticmethod
    def relBIAS(u1,u2):
        s = np.abs(u1).max()
        return (u1-u2).mean()/s
    
    @staticmethod
    def MAX(u1,u2):
        return np.abs(u2-u1).max()
    

class Evaluator:    
    def __init__(self,data,setup,selector=dict(),prefix=None):
        
        self.filt  = Data_Manager.data_filter(data,selector)
        self.data = data[self.filt]
        self.Energy = self.data['Energy'].to_numpy()
        self.setup = setup
        #if prefix is not None:
        #    fn = '{:s}_evaluations.log'.format(prefix)
        #else:
        #    fn = 'evaluations.log'
        
        #fname = '{:s}/{:s}'.format(setup.runpath,fn)
        #self.eval_fname = fname
        #self.eval_file = open(fname,'w')
        return
    def __del__(self):
        #self.eval_file.close()
        return


class Interfacial_Evaluator(Evaluator):
    def __init__(self,data,setup,selector=dict(),prefix=None):
        super().__init__(data,setup,selector,prefix)
        self.Uclass = self.data['Uclass'].to_numpy()
        
    def compare(self,fun,col):
        un = np.unique(self.data[col])
        res = dict()
        fun = getattr(measures,fun)
        for u in un:
            f = self.data[col] == u
            u1 = self.Energy[f]
            u2 = self.Uclass[f]
            res[ '='.join((col,u)) ] = fun(u1,u2)
        return res
    
    def get_error(self,fun,colval=dict()):
        f = np.ones(len(self.data),dtype=bool)
        for col,val in colval.items():
            f = np.logical_and(f,self.data[col]==val)
           
        u1 = self.Energy[f]
        u2 = self.Uclass[f]
        try:
            return fun(u1,u2)#
        except:
            return None
    
    def make_evaluation_table(self,funs,cols,add_tot=True,save_csv=None):

        if not iterable(funs):
            funs = [funs]
        if not iterable(cols):
            cols = [cols]
        
        
        uncols = [np.unique(self.data[col]) for col in cols]
        sudoindexes = list(itertools.product(*uncols))
        ev = pd.DataFrame(columns=funs+cols)
        
        for ind_iter,index in enumerate(sudoindexes):
            for fn in funs:
                colval = dict()
                fun = getattr(measures,fn)
                for i,col in enumerate(cols):    
                    ev.loc[ind_iter,col] = index[i]
                    colval[col] = index[i]
                    
                ev.loc[ind_iter,fn] = self.get_error(fun,colval)
        if add_tot:
            for fn in funs:
                fun = getattr(measures,fn)
                ev.loc['TOTAL',fn] = self.get_error(fun)
        
        ev = ev[cols+funs]
        
        if save_csv is not None:
            ev.to_csv('{:s}/{:s}'.format(self.setup.runpath,save_csv))
        return ev
                
                
    def print_compare(self,fun,col):
        #try:
           #f = open(self.eval_fname,'a')
        #except IOError:
            #f = self.eval_file
        res = self.compare(fun,col)
        for k,v in res.items():
            pr = '{:s} : {:s} = {:4.3f}'.format(k,fun,v)
            print(pr)
            #f.write(pr+'\n')
        #f.flush()
        #f.close()
        return

    
    def plot_predict_vs_target(self,size=2.35,dpi=500,
                               xlabel=r'$E_{int}$', ylabel=r'$U$',
                               col1='Energy', col2='Uclass',title=None,
                               units =r'$kcal/mol$',
                               label_map=None,
                               path=None, fname='pred_vs_target.png',attrs=None,
                               save_fig=True,compare=None,diff=False,scale=1.05):
        make_dir(path)
        x_data = self.data[col1].to_numpy()
        y_data = self.data[col2].to_numpy()
        xlabel += ' (' +units+')'
        ylabel += ' (' +units+')'
        if path is None:
            path = self.setup.runpath
        col = self.data[compare]
        uncol = np.unique(col)
        if attrs is not None:
            uncol = np.array(attrs)
        colors = get_colorbrewer_colors(len(uncol))
        
        if not diff:
            fig = plt.figure(figsize=(size,size),dpi=dpi)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='minor',length=size)
            plt.tick_params(direction='in', which='major',length=2*size)
            perf_line = [x_data.min()*scale,x_data.max()*scale]
            plt.xticks(fontsize=3.0*size)
            plt.yticks(fontsize=3.0*size)
            if title is not None:
                plt.title(title,fontsize=4.0*size)
            if compare is None:
                plt.plot(x_data, y_data,ls='None',color='purple',marker='o',markersize=5,fillstyle='none')
            else:
                for i,c in enumerate(uncol):
                    f = col == c
                    if label_map is not None:
                        lbl = label_map[c]
                    else:
                        lbl = c
                    plt.plot(x_data[f], y_data[f],label=lbl,ls='None',color=colors[i],
                             marker='o',markersize=5/3.5*size,fillstyle='none')
            plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
            plt.xlabel(xlabel,fontsize=3.5*size)
            plt.ylabel(ylabel,fontsize=3.5*size)
            plt.legend(frameon=False,fontsize=3.0*size)
            if fname is not None:
                plt.savefig(path +'\\'+fname,bbox_inches='tight')
            plt.show()
        else:
            for i,c in enumerate(uncol):
                fig = plt.figure(figsize=figsize,dpi=dpi)
                plt.minorticks_on()
                plt.tick_params(direction='in', which='minor',length=length_minor)
                plt.tick_params(direction='in', which='major',length=length_major)
                perf_line = [x_data.min()*1.05,x_data.max()*1.05]
                plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
                f = col == c
                plt.plot(x_data[f], y_data[f],label=c,ls='None',color=colors[i],
                             marker='o',markersize=5/3.5*sizew,fillstyle='none')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend(frameon=False)
                if fname is not None:
                    c = fname.split('.')[0]+c+fname.split('.')[-1]
                    plt.savefig(path+'\\'+c,bbox_inches='tight')
                plt.show()
        
        
        return
    
    def plot_superPW(self,PWparams,model,size=3.3,fname=None,
                       dpi=300,rmin= 1,rmax=5,umax=None,xlabel=None):
        
        figsize = (size,size) 
        fig = plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=5)
        plt.tick_params(direction='in', which='major',length=10)
        types = PWparams[list(PWparams.keys())[0]].index
        pw_u = {t:0 for t in types}
        pw_r = {k:0 for k in PWparams}
        r = np.arange(rmin+1e-9,rmax,0.01)
        for k,params in PWparams.items():
            pars = self.setup.model_parameters(model,params.columns)
            f = globals()['u_'+model]        
            for t, pot in params.iterrows():
                args = np.array([pot[p] for p in pars])
                u = f(r,args)
                pw_u[t]+=u.copy()
                
                if umax is not None:
                    filt = u<=umax
                    uf = u[filt]
                    rf = r[filt]
                else:
                    uf = u
                    rf =r 
                plt.plot(rf,uf,label='{}-{}'.format(k,t),ls='-',lw=0.6)
        for t in types:
            u = pw_u[t]
            if umax is not None:
                filt = u<=umax
                uf = u[filt]
                rf = r[filt]
            else:
                uf = u
                rf =r 
            plt.plot(rf,uf,label='{}-{}'.format('super',t),ls='--',lw=1.0)
        plt.legend(frameon=False,fontsize=1.5*size)
        if xlabel is None:
            plt.xlabel(r'r / $\AA$')
        else:
            plt.xlabel(xlabel)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.ylabel(r'$U_{'+'{:s}'.format(model)+'}$ / kcal/mol')
        plt.show()
        return 
    
    def plot_potential(self,params,model,size=3.5,fname=None,
                       dpi=500,rmin= 1,rmax=5,
                       umax=None,xlabel=None,ylabel=None,
                       title=None):
        
        pars = self.setup.model_parameters(model,params.columns)
        f = globals()['u_'+model]        
        colors = ['#d7191c','#fdae61','#abd9e9','#2c7bb6']
        figsize = (size,size) 
        fig = plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size)
        plt.tick_params(direction='in', which='major',length=2*size)
        lst = ['-','--','-.',(0,(1,1))]
        plt.xticks(fontsize=3.0*size)
        plt.yticks(fontsize=3.0*size)
        if title is not None:
            plt.title(title,fontsize=size*4.0)
        for i,(j, pot) in enumerate(params.iterrows()):
            try:
                rmx = rmax[j]
            except TypeError:
                rmx = rmax
            r = np.arange(rmin+1e-9,rmx,0.01)
            args = np.array([pot[p] for p in pars])

            u = f(r,args)
            if umax is not None:
                filt = u<=umax
                u = u[filt]
                r = r[filt]
            label = r'$\alpha\beta={:s}$ ${:s}$'.format(j[0],j[1])
            plt.plot(r,u,label=label,lw=size/2.0,color=colors[i],ls=lst[i])
        plt.legend(frameon=False,fontsize=2.5*size)
        if xlabel is None:
            plt.xlabel(r'r / $\AA$',fontsize=3.5*size)
        else:
            plt.xlabel(xlabel,fontsize=3.5*size)
        if ylabel is not None:
            plt.ylabel(ylabel,fontsize=3.5*size)
        else:    
            plt.ylabel(r'$U_{'+'{:s}'.format(model)+'}$ / kcal/mol',fontsize=3.5*size)
        
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        
        
        plt.show()
        return 
    
    
    def plot_scan_paths(self,size=2.65,dpi=300,
                   xlabel=r'scanning distance / $\AA$', ylabel=r'$E_{int}$ / $kcal/mol$',
                   title=None,show_fit=True,
                   path=None, fname=None,markersize=0.7,
                   n1=3,n2=2,maxplots=None,
                   selector=dict(),x_col=None,subsample=None,scan_paths=(0,1e15)):
        #make_dir(path)
        self.data.loc[:,'scanpath'] = ['/'.join(x.split('/')[:-2]) for x in self.data['filename']]
        
        filt = Data_Manager.data_filter(self.data,selector)
        data = self.data[filt]
        data = data[['scanpath','filename','Energy','scan_val','Uclass']]
        
        
        unq = np.unique(data['scanpath'])
        nsubs=n1*n2
        if maxplots is None:
            nu = len(unq)
        else:
            nu = min(int(len(unq)/nsubs)+1,maxplots)*nsubs
        figsize=(size,size)
        cmap = matplotlib.cm.get_cmap('tab20')
        
        if abs(n1-n2)!=1:
            raise Exception('|n1-n2| should equal one')
        for jp in range(0,nu,nsubs):
            fig = plt.figure(figsize=figsize,dpi=dpi)
            plt.xlabel(xlabel, fontsize=2.5*size,labelpad=4*size)
            plt.ylabel(ylabel, fontsize=2.5*size,labelpad=4*size)
            plt.xticks([])
            plt.yticks([])
            nfig = int(jp/nsubs)
            if title is None:
                fig.suptitle('set of paths {}'.format(nfig),fontsize=3*size)
            else:
                fig.suptitle(title,fontsize=3*size)
            gs = fig.add_gridspec(n1, n2, hspace=0, wspace=0)
            ax = gs.subplots(sharex='col', sharey='row')

            for j,fp in enumerate(unq[jp:jp+nsubs]):
                
                dp = data[ data['scanpath'] == fp ]
                x_data = dp['scan_val'].to_numpy()
                xi = x_data.argsort()   
                x_data = x_data[xi]
                c = cmap(j%nsubs/nsubs)
                e_data = dp['Energy'].to_numpy()[xi]
                u_data = dp['Uclass'].to_numpy()[xi]
                j1 = j%n1
                j2 = j%n2
                ax[j1][j2].plot(x_data, e_data, ls='none', marker='o',color=c,
                    fillstyle='none',markersize=markersize*size,label=r'$E_{int}$')
                if show_fit:
                    ax[j1][j2].plot(x_data, u_data, lw=0.3*size,color='k',label=r'fit')
                #ax[j1][j2].legend(frameon=False,fontsize=2.0*size)
                ax[j1][j2].minorticks_on()
                ax[j1][j2].tick_params(direction='in', which='minor',length=size*0.8,labelsize=size*2)
                ax[j1][j2].tick_params(direction='in', which='major',length=size*1.4,labelsize=size*2)
            
        #plt.legend(frameon=False)
            if path is None:
                path= self.setup.runpath
                
            if fname is None:
                plt.savefig('{}/spths{}.png'.format(path,nfig),bbox_inches='tight')
            else:
                plt.savefig('{:s}/{:s}'.format(path,fname))
            plt.show()
    
    def plot_eners(self,figsize=(3.3,3.3),dpi=300,
                   xlabel=r'$\AA$', ylabel=r'$kcal/mol$',
                   col1='Energy', col2='Uclass',title=None,
                   length_minor=5, length_major=10,
                   path=None, fname=None,
                   selector=dict(),x_col=None,subsample=None):
        #make_dir(path)
        data = self.data
        if col1=='Energy':
            lab1 = r'$E_{dft}$'
        else:
            lab1 = col1
        if col2 =='Uclass':
            lab2 = r'$U_{vdw}$'
        else:
            lab2 = col2
        if path is None:
            path = self.setup.runpath
        filt = Data_Manager.data_filter(data,selector)
        
        if x_col is not None:
            x_data = data[x_col][filt].to_numpy()
        else:
            x_data = np.array(data.index)
            xlabel = 'index'
        
        xi = x_data.argsort()
        if subsample is not None:
            xi = np.random.choice(xi,replace=False,size=int(subsample*len(xi)))
            xi.sort()
            
        x_data = x_data[xi]
        e_data = data[col1][filt].to_numpy()[xi]
        u_data = data[col2][filt].to_numpy()[xi]
        
        fig = plt.figure(figsize=figsize,dpi=dpi)
        if title is not None:
            plt.title(title)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=length_minor)
        plt.tick_params(direction='in', which='major',length=length_major)
        plt.plot([x_data.min(),x_data.max()],[0,0],lw=0.5,ls='--',color='k')
        plt.plot(x_data, e_data,label=lab1,ls='None',color='red',
                 marker='o',markersize=2,fillstyle='none')
        plt.plot(x_data, u_data,label=lab2,ls='None',color='blue',
                 marker='o',markersize=2,fillstyle='none')
        for i in range(len(x_data)):
            x = [x_data[i],x_data[i]]
            y = [e_data[i],u_data[i]]
            plt.plot(x,y,ls='-',lw=0.2,color='k')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        if fname is not None:
            plt.savefig(path +'\\'+fname,bbox_inches='tight')
        plt.show()

 ####
##### ##### ##
class mappers():
    elements_mass = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}
    @property
    def atomic_num(self):
        return {'{:d}'.format(int(i+1)):elem for i,elem in enumerate(self.elements_mass.keys())}
    
    
def demonstrate_bezier(by=None,dpi=300,size=3.2,fname=None,format='eps',
                      rhomax=10.0,show_points=True,seed=None):
    if seed is not None:
        np.random.seed(seed)
    if by is None:
        y = np.array([0,0,5.0,-12.0,-18.0,23,0,0])
        y1 = y.copy()
        y1[2:-2] += np.random.normal(0,5.0,4)
        y2 = y.copy() +np.random.normal(0,5.9,8)
        by = [y,y1,y2]
    else:
        if type(by) is not list:
            by = [by,]
    n = by[0].shape[0]
    dx = rhomax/(n-1)
    bx = np.array([i*dx for i in range(n)])
    curve = globals()['u_bezier']
    drho=0.01
    rh = np.arange(drho,rhomax+drho,drho)
    colors = ['#d7191c','#fdae61','#abd9e9','#2c7bb6']
    figsize = (size,size) 
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length=5)
    plt.tick_params(direction='in', which='major',length=10)
    markers =['o','s','x','v']
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Illustration of the Bezier curves',fontsize=3.4*size)
    for j in range(len(by)):
        y= by[j]
        u = curve(rh,y)
        plt.plot(rh,u,color=colors[j],lw=0.6*size,label='curve {:d}'.format(j+1))
        if show_points:
            plt.plot(bx,y,ls='none',marker=markers[j],markeredgewidth=0.5*size,
                     markersize=2.0*size,fillstyle='none',color=colors[j])
    plt.legend(frameon=False,fontsize=2.5*size)
    if fname is not None:
         plt.savefig(fname,bbox_inches='tight',format=format)
    plt.show()
    
    return