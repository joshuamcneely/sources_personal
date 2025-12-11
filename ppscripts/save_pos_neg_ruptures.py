#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.ndimage import convolve
from scipy.interpolate import interp1d

import ifasha.datamanager as idm  
import ppscripts.postprocess as pp

from ppscripts.save_ruptures import get_is_sticking

rpt_default_fname = 'pos_neg_ruptures.pkl'

# -----------------------------------------------------------------------------
def get_pos_neg_ruptures(sname, **kwargs):
    """
returns 

[time, pos_front_x, pos_front_z] (Nt, Nt x Nx, Nt x Nx)
[time, neg_front_x, neg_front_z] (Nt, Nt x Nx, Nt x Nx)
    """

    d_slip = kwargs.get('d_slip', 0)
    d_slip = float(d_slip)    

    wdir = kwargs.get('wdir', './data')
    group = kwargs.get('group','interface')
    sname = pp.sname_to_sname(sname)

    # get data
    dma = idm.DataManagerAnalysis(sname,wdir)
    data = dma(group)

    dim = 2
    if data.has_field(idm.FieldId('position',2)) or data.has_field(idm.FieldId('coord',2)): 
        dim =3
    if data.has_field(idm.FieldId('position',0)):
        posfld='position'
    elif data.has_field(idm.FieldId('coord',0)):
        posfld='coord'
    else:
        raise('Does not have any position information')

    # position
    if dim == 2:
        position_x = data.get_field_at_t_index(idm.FieldId(posfld,0),0) # position at beginning
    else:
        position_x = data.get_field_at_t_index(idm.FieldId(posfld,0),0) # position at beginning      
        position_z = data.get_field_at_t_index(idm.FieldId(posfld,2),0) # position at beginning      
    is_sticking = get_is_sticking(data,d_slip)
    #plt.pcolormesh(is_sticking)
    #plt.show()
    print(np.max(is_sticking),np.min(is_sticking))
    time = data.get_full_field(idm.FieldId('time'))
    print(time.shape)

    if posfld=='coord':
        nx = len(np.unique(position_x))
        nz = len(np.unique(position_z))
        PX = position_x[0].reshape(nx,nz)
        PZ = position_z[0].reshape(nx,nz)
        STICK = np.array([s.reshape(nx,nz) for s in is_sticking])
    else:
        raise RuntimeError('reshaping not implemeted')
    
    # x_coords = PX[:,0]
    # z_coords = PZ[0,:]

    pos_front_x_at_t = []
    pos_front_z_at_t = []
    neg_front_x_at_t = []
    neg_front_z_at_t = []
    for t in range(len(time)):

        S = STICK[t]

        front_x=[]
        front_z=[]
        # find contours
        # Sobel conv kernel to detect edges
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Gy = np.array([[ 1, 2, 1],
                       [ 0, 0, 0],
                       [-1,-2,-1]])
        
        edgex = convolve(S,Gx)
        edgey = convolve(S,Gy)
        edge = edgex + edgey
        
        # front in positive x direction
        pos_idx = edge > 0
        # front in negative x direction
        neg_idx = edge < 0
        for dir, idx in zip([1,-1],[pos_idx,neg_idx]):
            fx = PX[idx]
            fz = PZ[idx]
            idx = np.argsort(fz)
            fx = fx[idx]
            fz = fz[idx]
            if dir==1:
                pos_front_x_at_t.append(fx)
                pos_front_z_at_t.append(fz)
            else:
                neg_front_x_at_t.append(fx)
                neg_front_z_at_t.append(fz)

    return [time,np.array(pos_front_x_at_t),np.array(pos_front_z_at_t)],[time,np.array(neg_front_x_at_t),np.array(neg_front_z_at_t)]

# -----------------------------------------------------------------------------
def clean_front(t,fxt,fzt,show_fig=False):
    """
    get unique x and z positions

    return 
    #    t (Nt)
    x_pos (Nt x Nz)
    z_pos (Nt x Nz)

    """
    # for plot
    l=0
    cmap = plt.cm.get_cmap("viridis")

    z = []
    for fz in fzt:
        z+=list(fz)
    zcoord=np.sort(np.unique(z)) 
    xnew=[]
    znew=[]
    for ti,fx,fz,c in zip(t,fxt,fzt,cmap(np.arange(len(t))/len(t))):

        # filter out duplicated x positions for a given z
        # use max instead of average to get the upper envelope
        zflt = []
        xflt = []
        for z in np.unique(fz):
            zflt.append(z)
            idx = fz==z
            #xflt.append(np.average(fx[idx]))
            xflt.append(np.min(fx[idx]))
            
        # filter out duplicated z positions for same x
        xu = []
        zu = []
        for i in range(len(zflt)-1):
            if xflt[i]!=xflt[i+1]:
                if 1:
                    xu.append(min(xflt[i],xflt[i+1]))
                    zu.append(0.5*(zflt[i]+zflt[i+1]))
                
                else:
                    if 1:# xflt[i]!=xflt[i-1]:
                        xu.append(xflt[i])
                        zu.append(zflt[i])
                    xu.append(xflt[i+1])
                    zu.append(zflt[i+1])

            elif i==0:
                xu.append(xflt[i])
                zu.append(zflt[i])
            elif  i==len(zflt)-2:
                xu.append(xflt[i+1])
                zu.append(zflt[i+1])

        # interpolate to original grid
        zn = zcoord
        xn = interp1d(zu,xu,fill_value=np.nan,bounds_error=False)(zn)
        znew.append(zn)
        xnew.append(xn)

        if show_fig:
            
            l+=1
            if l%10: 
                continue

            plt.plot(fz,fx,'.-',color=c,label=ti)
            plt.plot(zflt,xflt,'r.-')

            plt.plot(zn,xn,'k.-')
            plt.plot(zu,xu,'g.-')
    plt.legend(loc='best')
    plt.xlabel('z')
    plt.ylabel('x')
    #plt.show()

    return np.array(xnew), np.array(znew)

# -----------------------------------------------------------------------------
def detect_tangential_front(front,dt,dx,dz):
    t,x,z = front

    t_final = []
    xmin  = []
    xmax  = []
    xmean = []
    z_tpos = []
    z_tneg = []
    znuc = None
    for ti,xi,zi in zip(t,x,z):
        
        if len(xi)==0: 
            print("skip initial phase empty array")
            continue

        if np.max(xi)-np.min(xi)<2*dx: 
            print("skip initial phase")
            continue


        xm = np.average([np.min(xi),np.max(xi)])

        if len(xmean) and xm < np.min(xmean)+dx:
            continue

        if znuc is None:
            znuc = np.average(zi[xi==np.max(xi)])
            zp = znuc
            zn = znuc
            print('znuc',znuc)
        else:
            if len(xi[zi>znuc-dz])<2 or len(xi[zi<znuc+dz])<2:
                continue

            zp = interp1d(xi[zi>znuc],zi[zi>znuc])(xm)
            zn = interp1d(xi[zi<znuc],zi[zi<znuc])(xm)

        t_final.append(ti)
        z_tpos.append(zp)
        z_tneg.append(zn)

        xmin.append(np.min(xi))
        xmax.append(np.max(xi))
        xmean.append(xm)
    t_final,xmean,z_tpos,z_tneg = [ np.array(t_final),np.array(xmean),np.array(z_tpos),np.array(z_tneg)]
    return t_final,xmean,z_tpos,z_tneg

    
# -----------------------------------------------------------------------------
def save_pos_neg_ruptures(snames,**kwargs):

    wdir = kwargs.get('wdir','./data')

    
    for sname in snames:
        sname = pp.sname_to_sname(sname)
        fname = kwargs.get('rpt_fname',rpt_default_fname)

        # get ruptures
        ruptures = get_pos_neg_ruptures(sname, **kwargs)

        # save to temporary file
        tmp_fname = 'save_ruptures.tmp'
        with open(tmp_fname, 'wb') as fl:
            pickle.dump(ruptures, fl)
        print('save '+fname)
        # copy temporary file into datamanager
        dm = idm.DataManager(sname,wdir)
        dm.add_supplementary(fname,tmp_fname,True)

# ------------------------------------------------------------------------------
def load_pos_neg_ruptures(sname,**kwargs):

    sname = pp.sname_to_sname(sname)

    # use data manager to load ruptures
    wdir = kwargs.get('wdir','./data')
    fname = kwargs.get('rpt_fname',rpt_default_fname)
    dm = idm.DataManager(sname,wdir)

    print(fname)
    try:
        open(dm.get_supplementary(fname),'rb')
        print('found')
    except:
        save_ruptures([sname], **kwargs)
        dm = idm.DataManager(sname,wdir)
        
    with open(dm.get_supplementary(fname),'rb') as fl:
        ruptures = pickle.load(fl)

    return ruptures


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    #if len(sys.argv) < 2:
    #    sys.exit('Missing argument! usage: ./save_ruptures.py simid[s]')

    # simid can be 22,24,30-33 -> [22,24,30,31,32,33]
    #simids = pp.string_to_simidlist(str(sys.argv[1]))
    simids =['stick_break_3d_{}_from2d_restarted'.format(i) for i in [
        31]]
    
    #try:
    rpts = load_pos_neg_ruptures(simids[0])
    #except:
    #    save_pos_neg_ruptures(simids)
    
    pos_front,neg_front=rpts

    cmap = plt.cm.get_cmap("viridis")
    
    t,fxt,fzt = pos_front
    fxu,fzu = clean_front(t,fxt,fzt,show_fig=True)#False)
    plt.show()
    #continue
    i=0
    lst='-'
    t -= t[0]

    input_data = pp.get_input_data(simids[0])
    dx=input_data['x_length']/input_data['nb_x_elements']
    dz=input_data['z_length']/input_data['nb_z_elements']
    dt=input_data['dump_interval']

    t_final,xmean,z_tpos,z_tneg = detect_tangential_front(pos_front,dt,dx,dz)

    for ti,fx,fz,fxr,fzr,c in zip(t,
                                  fxu,fzu,
                                  fxt,fzt,
                                  cmap(np.arange(len(t))/len(t))):
        i+=1
        if i%10:continue
            
        #plt.plot(fzr,fxr,lst,color=c,label=ti)
        plt.plot(fz,fx,lst,color=c,label=ti) 
    plt.plot(z_tpos,xmean,'k.-')
    plt.plot(z_tneg,xmean,'k.-')

    plt.legend(loc='best')
    plt.xlabel('z')
    plt.ylabel('x')
    
    plt.show()



