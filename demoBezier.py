# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:03:52 2023

@author: n.patsalidis
"""

def demonstrate_bezier(dpi=300,size=3.2,fname=None,format='.png',
                      rhomax=10.0,show_points=True,seed=None,illustration='y'):
    if seed is not None:
        np.random.seed(seed)
    
    y = np.array([10.0,0,0,5.0,-12.0,-18.0,23,0,0])
    y1 = y.copy()
        
    y1[3:-2] += np.random.normal(0,5.0,4)
    y2 = y.copy() 
    y2[0] = 13.0
    y3 = y.copy()
    y3[0] = 7.0
    data = {0:[y,y1],1:[y,y2,y3]}
    n = by[0].shape[0]
    dx = rhomax/(n-1)
    bx = np.array([i*dx for i in range(n)])
    curve = globals()['u_bezier']
    drho=0.01
    
    colors = ['#1b9e77','#d95f02','#7570b3']
    figsize = (size,size) 
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.minorticks_on()
    markers =['o','s','x','v']
    plt.xlabel(r'$\mathbf{x}$')
    plt.ylabel(r'$\mathbf{y}$')
    plt.xticks([])
    plt.yticks([])
    gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
    ax = gs.subplots(sharex='col', sharey='row')
    ax[0].title(r'Varying the "$\mathbf{y}$" position of the Bezier CPs',fontsize=2.5*size)
    ax[1].title(r'Varying the "$\mathbf{x}$" positions of the Bezier CPs ',fontsize=2.5*size)
    
    for i,(k,d) in data.items():
        
        ax[i].tick_params(direction='in', which='minor',length=size)
        ax[i].tick_params(direction='in', which='major',length=2*size)
        for y in d:
            rh = np.arange(drho,y[0]+drho,drho)
            u = curve(rh,y)
            label=None
            ax[i].plot(rh,u,color=colors[j],lw=0.6*size,label=label)
            if show_points:
                bx = np.arange(0,y[0]+0.01,y[0]/(y.shape[0]-2))
                ax[i].plot(bx,y[1:],ls='none',marker=markers[j],markeredgewidth=0.5*size,
                     markersize=2.0*size,fillstyle='none',color=colors[j])

        ax[i].legend(frameon=False,fontsize=2.5*size)
    if fname is not None:
         plt.savefig(fname,bbox_inches='tight',format=format)
    plt.show()
    
    return