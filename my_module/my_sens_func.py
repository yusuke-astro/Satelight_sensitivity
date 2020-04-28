import  numpy as np
from scipy import integrate

GHz = 1e9
p = 1e-12
W = 0.0883*p
u = 1e-6
m = 1e-3
e_0 = 8.85418782*10**(-12)
h = 6.62607004*10**(-34)
k_B = 1.38064852*10**(-23)
c = 299792458

def width(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f
    #fracはtable2,3,4 in page 9,11,12にある，center of frequencyに対する積分のバンド端を定義するために使う値です．
    if f == 40*GHz:
        frac = 0.30
    if f == 50*GHz:
        frac = 0.30
    if f == 60*GHz:
        frac = 0.23
    if f == 68*GHz:
        frac = 0.23
    if f == 78*GHz:
        frac = 0.23
    if f == 89*GHz:
        frac = 0.23
    if f == 100*GHz:
        frac = 0.23
    if f == 119*GHz:
        frac = 0.30
    if f == 140*GHz:
        frac = 0.30
    if f == 166*GHz:
        frac = 0.30
    if f == 195*GHz:
        frac = 0.30
    if f == 235*GHz:
        frac = 0.30
    if f == 280*GHz:
        frac = 0.30
    if f == 337*GHz:
        frac = 0.30
    if f == 402*GHz:
        frac = 0.23

    #BW(band_width)はCenter of frequencyとfracで定義される積分範囲の半分で，実際に積分するときは
    #freq-BW → freq+BWの範囲で積分を行います．
    BW = (f * frac)/2.0
    return BW

def r_HWP_LFT(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    if f == 40*GHz:
        r_HWP = 0.1000
    if f == 50*GHz:
        r_HWP = 0.1000
    if f == 60*GHz:
        r_HWP = 0.0682
    if f == 68*GHz:
        r_HWP = 0.0514
    if f == 78*GHz:
        r_HWP = 0.0415
    if f == 89*GHz:
        r_HWP = 0.0368
    if f == 100*GHz:
        r_HWP = 0.0275
    if f == 119*GHz:
        r_HWP = 0.013
    if f == 140*GHz:
        r_HWP = 0.0014
    return r_HWP

def epsi_HWP_LFT(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B

    if f == 40*GHz:
        epsi_HWP = 0.0038
    if f == 50*GHz:
        epsi_HWP = 0.0048
    if f == 60*GHz:
        epsi_HWP = 0.0058
    if f == 68*GHz:
        epsi_HWP = 0.0066
    if f == 78*GHz:
        epsi_HWP = 0.0075
    if f == 89*GHz:
        epsi_HWP = 0.0097
    if f == 100*GHz:
        epsi_HWP = 0.0115
    if f == 119*GHz:
        epsi_HWP = 0.0135
    if f == 140*GHz:
        epsi_HWP = 0.0161
    return epsi_HWP

def ita_HWP_LFT(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f
    if f == 40*GHz:
        ita_HWP = 0.8962
    if f == 50*GHz:
        ita_HWP = 0.8952
    if f == 60*GHz:
        ita_HWP = 0.9260
    if f == 68*GHz:
        ita_HWP = 0.9420
    if f == 78*GHz:
        ita_HWP = 0.9510
    if f == 89*GHz:
        ita_HWP = 0.9535
    if f == 100*GHz:
        ita_HWP = 0.9610
    if f == 119*GHz:
        ita_HWP = 0.9735
    if f == 140*GHz:
        ita_HWP = 0.9825
    return ita_HWP

def ita_Apt_LFT(f, D):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f
    ita_Apt = 1.0 - np.exp(-((np.pi)**2 / 2.0) * ((D*m) / (2.75 * 3.0 * WL))**2)
    return ita_Apt

def ita_20K(f,D):#freqに対して値が二つある．table29
    global GHz
    #global D
    #print("func__ita20K")
    #print("D=",D)
    if f == 40*GHz:
        ita20K = 0.0130
    if f == 50*GHz:
        ita20K = 0.0084
    if f == 60*GHz:
        ita20K = 0.0049
    if f == 68*GHz:
        if D == 24.0:
            ita20K = 0.0030
        if D == 16.0:
            ita20K = 0.0103
    if f == 78*GHz:
        if D == 24.0:
            ita20K = 0.0015
        if D == 16.0:
            ita20K = 0.0075
    if f == 89*GHz:
        if D == 24.0:
            ita20K = 0.0006
        if D == 16.0:
            ita20K = 0.0051
    if f == 100*GHz:
        ita20K = 0.0033
    if f == 119*GHz:
        ita20K = 0.0013
    if f == 140*GHz:
        ita20K = 0.0004
    return ita20K

def epsi_Mir(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    e_Mir = 4.0 * np.sqrt(np.pi * f * e_0 * 1.39*10**(-8))
    return e_Mir

def r_Mir(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f

    r_Mir = 1.0 - np.exp(-((4.0 * np.pi * 2.0*u) / WL)**2)
    return r_Mir

def ita_Mir(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B

    ita_Mir = 1.0 - epsi_Mir(f) - r_Mir(f)
    return ita_Mir

def epsi_2KF(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f

    e_2K = 1.0 - np.exp((-2.0 * np.pi * 5.0*m * 1.5 * 2.3*10**(-4)) / WL)
    return e_2K

def r_2KF():
    r_2K = 0.05
    return r_2K

def ita_2K(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f

    ita_2KF = 1.0 - r_2KF() - epsi_2KF(f)
    return ita_2KF

def ita_5K(f, D):
    return 1.0-ita_Apt_LFT(f, D)

def epsi_Lenslet(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f
    epsi_Lens = 1.0 - np.exp((-2.0 * np.pi * 9.0*m * 3.4 * 5.0*10**(-5)) / WL)
    return epsi_Lens

def r_Lenslet():
    r_lens = 0.05
    return r_lens

def ita_Lenslet(f):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    WL = c/f
    ita_Lens = 1.0 - r_Lenslet() - epsi_Lenslet(f)
    return ita_Lens


def r_det_LFT():
    r_det = 0.32
    return r_det

def ita_det_LFT():
    ita_det = 1.0 - r_det_LFT()
    return ita_det

def Bose(f,T):
    global GHz
    global p
    global c
    global u
    global m
    global e_0
    global h
    global k_B
    global F
    bose = (h * f) / (np.exp((h * f) / (k_B * T)) - 1.0)
    return bose
