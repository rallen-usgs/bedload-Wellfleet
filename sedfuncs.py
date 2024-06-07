import numpy as np

def qkhfs( w, h ):
    """
    Quick iterative calculation of kh in gravity-wave dispersion relationship
    kh = qkhfs(w, h )
    
    Input
        w - angular wave frequency = 2*pi/T where T = wave period [1/s]
        h - water depth [m]
    Returns
        kh - wavenumber * depth [ ]

    Orbital velocities from kh are accurate to 3e-12 !

    RL Soulsby (2006) \"Simplified calculation of wave orbital velocities\"
    HR Wallingford Report TR 155, February 2006
    Eqns. 12a - 14
    """
    g = 9.81
    x = w**2.0 *h/g
    y = np.sqrt(x) * (x<1.) + x *(x>=1.)
    # is this faster than a loop?
    t = np.tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = np.tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    t = np.tanh( y )
    y = y-( (y*t -x)/(t+y*(1.0-t**2.0)))
    kh = y
    return kh


def ub_func (T, k, H, h):
    """
    ub_func  Calculate near-bottom wave-orbital velocity amplitude

    ub = ub_f(T, k, H, h)

    Input:
    T  = sig. wave period (s)
    H  = sig. wave height ( = 2 * amplitude of surface wave; m
    h  = water depth (m)
    k  = wave number (use waven or qkhf)
    Returns:
    ub = Hs*pi / T*sinh(kh)

    Chris Sherwood, USGS
    Converted to python 
    cf. Dyer(1986) eqn. 3.50, p. 98
    or  Komar(1976) p. 42
    """
    w = (2. * np.pi)/T
    kh = k*h
    amp = H / (2. * np.sinh(kh))
    ub = w * amp
    return ub


def m94( ubr, wr, ucr, zr, phiwc, kN, iverbose=False ):
    """
    M94 - Grant-Madsen model from Madsen(1994)
    ustrc, ustrr, ustrm, dwc, fwc, zoa =
        m94( ubr, wr, ucr, zr, phiwc, kN, iverbose )

    Input:
        ubr = rep. wave-orbital velocity amplitude outside wbl [m/s]
        wr = rep. angular wave frequency = 2pi/T [rad/s]
        ucr = current velocity at height zr [m/s]
        zr = reference height for current velocity [m]
        phiwc = angle between currents and waves at zr (radians)
        kN = bottom roughness height (e.q. Nikuradse k) [m]
        iverbose = True/False; when True, extra output
    Returned as tuple
        ustrc  = current friction velocity         u*c [m/s]
        ustrr  = w-c combined friction velocity    u*r [m/s]
        ustrwm = wave max. friction velocity      u*wm [m/s]
        dwc = wave boundary layer thickness [m]
        fwc = wave friction factor [ ]
        zoa = apparent bottom roughness [m]
        
    Chris Sherwood, USGS
    November 2005: Removed when waves == 0
    July 2005: Removed bug found by JCW and RPS
    March 2014: Ported from Matlab to Python
    """
    MAXIT = 20
    vk = 0.41
    rmu=np.zeros((MAXIT,1))
    Cmu=np.zeros((MAXIT,1))
    fwci=np.zeros((MAXIT,1))
    dwci=np.zeros((MAXIT,1))
    ustrwm2=np.zeros((MAXIT,1))
    ustrr2=np.zeros((MAXIT,1))
    ustrci=np.zeros((MAXIT,1))

    # ...junk return values
    ustrc = 99.99
    ustrwm = 99.99
    ustrr = 99.99
    fwc = .4
    zoa = kN/30.
    zoa = zoa
    dwc = kN

    # ...some data checks
    if( wr <= 0. ):
        print('WARNING: Bad value for frequency in M94: wr={0}\n'.format(wr))
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa
	
    if( ubr < 0. ):
        print('WARNING: Bad value for orbital vel. in M94: ub={0}\n'.format(ubr))
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa

    if( kN < 0. ):
        print('WARNING: Weird value for roughness in M94: kN={0}\n'.format(kN))
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa
	
    if( (zr<zoa or zr<0.05) and iverbose == True):
        print('WARNING: Low value for ref. level in M94: zr={0}\n'.format(zr))	

    zo = kN/30.
    if(ubr <= 0.01):
        if(ucr <= 0.01):
            # ...no waves or currents
            ustrc = 0.
            ustrwm = 0.
            ustrr = 0.
            return ustrc, ustrr, ustrwm, dwc, fwc, zoa
        # ...no waves
        ustrc = ucr * vk / log(zr/zo) 
        ustrwm = 0.
        ustrr = ustrc
        return ustrc, ustrr, ustrwm, dwc, fwc, zoa
  
    cosphiwc =  np.abs(np.cos(phiwc))
    rmu[0] = 0.
    Cmu[0] = 1.
    cukw = Cmu[0]*ubr/(kN*wr)
    print(Cmu[0], cukw)
    fwci[0] = fwc94( Cmu[0], cukw )                 #Eqn. 32 or 33
    ustrwm2[0]= 0.5*fwci[0]*ubr*ubr                 #Eqn. 29
    ustrr2[0] = Cmu[0]*ustrwm2[0]                   #Eqn. 26
    ustrr = np.sqrt( ustrr2[0] )
    dwci[0] = kN
    if (cukw >= 8.):
        dwci[0]= 2.*vk*ustrr/wr
    lnzr = np.log(zr/dwci[0])
    lndw = np.log(dwci[0]/zo)
    lnln = lnzr/lndw
    bigsqr = (-1.+np.sqrt(1+ ((4.*vk*lndw)/(lnzr*lnzr))*ucr/ustrr))
    ustrci[0] = 0.5*ustrr*lnln*bigsqr
    nit = 1

    for i in range(1,MAXIT):      
        rmu[i] = ustrci[i-1]*ustrci[i-1]/ustrwm2[i-1]
        Cmu[i] = np.sqrt(1.+2.*rmu[i]*cosphiwc+rmu[i]*rmu[i]) #Eqn 27
        cukw = Cmu[i]*ubr/(kN*wr)
        fwci[i] = fwc94( Cmu[i], cukw )               #Eqn. 32 or 33
        ustrwm2[i]= 0.5*fwci[i]*ubr*ubr               #Eqn. 29
        ustrr2[i] = Cmu[i]*ustrwm2[i]                 #Eqn. 26
        ustrr = np.sqrt( ustrr2[i] )
        dwci[i] = kN
        if ((Cmu[i]*ubr/(kN*wr))>= 8.):
            dwci[i]= 2.*vk*ustrr/wr                   #Eqn.36
        lnzr = np.log( zr/dwci[i] )
        lndw = np.log(dwci[i]/zo)
        lnln = lnzr/lndw
        bigsqr = (-1.+np.sqrt(1+ ((4.*vk*lndw)/(lnzr*lnzr))*ucr/ustrr))
        ustrci[i] = 0.5*ustrr*lnln*bigsqr              #Eqn. 38
        diffw = abs( (fwci[i]-fwci[i-1])/fwci[i] )
        # print i,diffw
        if(diffw < 0.0005):
            break
        nit = nit+1
        ustrwm = np.sqrt( ustrwm2[nit] )
        ustrc = ustrci[nit]
        ustrr = np.sqrt( ustrr2[nit] )

    zoa = np.exp( np.log(dwci[nit])-(ustrc/ustrr)*np.log(dwci[nit]/zo) ) #Eqn. 11
    fwc = fwci[nit]
    dwc = dwci[nit]
    if(iverbose==True):
        print("M94 nit=",nit)
        for i in range(nit):
            print( \
            'i={0} fwc={1} dwc={2} u*c={3} u*wm={4} u*r={5}'\
	    """  """.format(i,fwci[i],dwci[i],ustrci[i],np.sqrt(ustrwm2[i]),np.sqrt(ustrr2[i])))

    return ustrc, ustrr, ustrwm, dwc, fwc, zoa


def fwc94( cmu, cukw ):
    """
    fwc94 - Wave-current friction factor
    fwc = fwc94( cmu, cukw )
    Equations 32 and 33 in Madsen, 1994

    csherwood@usgs.gov 4 March 2014
    """
    fwc = 0.00999 #meaningless (small) return value
    if( cukw <= 0. ):
        print('ERROR: cukw too small in fwc94: {0}\n'.format(cukw))
        return fwc

    if( cukw < 0.2 ):
        fwc = np.exp( 7.02*0.2**(-0.078) - 8.82 )
        print('WARNING: cukw very small in fwc94: {0}\n'.format(cukw))
    if( (cukw >= 0.2) and (cukw <= 100.) ):
        fwc = cmu*np.exp( 7.02*cukw**(-0.078)-8.82 )
    elif( (cukw > 100.) and (cukw <= 10000.) ):
        fwc = cmu*np.exp( 5.61*cukw**(-0.109)-7.30 )
    elif( cukw > 10000.):
        fwc = cmu*np.exp( 5.61*10000.**(-0.109)-7.30 )
        print('WARNING: cukw very large in fwc94: {0}\n'.format(cukw))

    return fwc
    
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length. 

    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas   
    Changed radius of earth to 6371. CRS
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371. * c
    return km


def get_bearing(lon1, lat1, lon2, lat2):
    #https://stackoverflow.com/questions/54873868/python-calculate-bearing-between-two-lat-long
    dLon = (lon2 - lon1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng_rad = np.arctan2(x,y)
    brng_deg = np.degrees(brng_rad)

    return brng_deg


def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial (km)
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return (np.degrees(lat2), np.degrees(lon2),)


def pcoord(x, y):
    """
    Convert x, y to polar coordinates r, az (geographic convention)
    r,az = pcoord(x, y)
    """
    r = np.sqrt(x**2 + y**2)
    az = np.degrees(np.arctan2(x, y))
    # az[where(az<0.)[0]] += 360.
    az = (az+360.)%360.
    return r, az


def xycoord(r, az):
    """
    Convert r, az [degrees, geographic convention] to rectangular coordinates
    x,y = xycoord(r, az)
    """
    x = r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))
    return x, y


def dist_bearing(lon1, lat1, lon2, lat2):
    dist = haversine(lon1, lat1, lon2, lat2)
    brng = get_bearing(lon1, lat1, lon2, lat2)
    return dist, brng