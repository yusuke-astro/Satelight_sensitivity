"""
このプログラムは長谷部さんのノートにより定義されたPoptをLFTが観測する各周波数に渡って積分して，長谷部さんの計算結果とプロットして比較する．
23行目のswitchを0 or 1にとることで，Pixel pich:Dを変更することができる．
"""

import  numpy as np
from scipy import integrate
from my_module import my_sens_func as st
import matplotlib.pyplot as plt

GHz = 1e9
p = 1e-12
Tera=1e12
W = 0.08883*p
u = 1e-6
m = 1e-3
e_0 = 8.85418782*10**(-12)
h = 6.62607004*10**(-34)
k_B = 1.38064852*10**(-23)
c = 299792458
############################################
switch = 0
#freqに対してDが2つあるところがある
def define_D(f):
    #周波数に対してDが2つ存在するところで，使用するDの値を24にするか16にするかを決める仮想パラメータ
    #switch=1なら78,68,89GHzにおいてD=24.0でswitch=0ならD=16.0が選択される
    global switch
    if f == 40*GHz:
        D = 24.0
    if f == 60*GHz:
        D = 24.0
    if f == 78*GHz:
        if switch == 1:
            D = 24.0
        if switch == 0:
            D = 16.0
    if f == 50*GHz:
        D = 24.0
    if f == 68*GHz:
        if switch == 1:
            D = 24.0
        if switch == 0:
            D = 16.0
    if f == 89*GHz:
        if switch == 1:
            D = 24.0
        if switch == 0:
            D = 16.0
    if f == 100*GHz:
        D = 16.0
    if f == 119*GHz:
        D = 16.0
    if f == 140*GHz:
        D = 16.0
    return D

def P_CMB(f):
    global F
    D = define_D(F)
    T = 2.725
    return st.ita_HWP_LFT(F)*st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_HWP(f):
    global F
    D = define_D(F)
    T = 20.00
    T_r = 5.000
    HWP_e = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_HWP_LFT(F)
    HWP_r = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_HWP_LFT(F)
    return HWP_e + HWP_r

def P_Ape(f):
    global F
    D = define_D(F)
    T = 5.000
    return st.ita_5K(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)

def P_m1(f):
    global F
    T = 5.000
    Pri = st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)
    Pri_e = Pri * st.epsi_Mir(f)
    Pri_r = Pri * st.r_Mir(f)
    return Pri_e + Pri_r

def P_m2(f):
    global F
    T = 5.000
    Sec = st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)
    Sec_e = Sec * st.epsi_Mir(f)
    Sec_r = Sec * st.r_Mir(f)
    return Sec_e + Sec_r

def P_20K(f):
    global F
    D = define_D(F)
    T = 20.0
    return st.ita_20K(F,D)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_2KF(f):
    global F
    T = 2.000
    T_r = 0.100
    Flt2K_e = st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_2KF(f)
    Flt2K_r = st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_2KF()
    return Flt2K_e + Flt2K_r

def P_Lens(f):
    global F
    T = 0.100
    T_r = 5.000
    Lens_e = st.ita_det_LFT()*st.Bose(f,T) * st.epsi_Lenslet(f)
    Lens_r = st.ita_det_LFT()*st.Bose(f,T_r) * st.r_Lenslet()
    return Lens_e + Lens_r

def P_Det(f):
    T = 0.100
    Det_e = st.Bose(f,T) * st.r_det_LFT()
    Det_r = st.Bose(f,T) * st.ita_det_LFT()
    return Det_e + Det_r

def P_Opt(f):
    return P_CMB(f)+P_HWP(f)+P_Ape(f)+P_m1(f)+P_m2(f)+P_20K(f)+P_2KF(f)+P_Lens(f)+P_Det(f)

#freq_note = np.array([40,60,78,50,68,89,119,100,140])
#hasebe = np.array([0.36,0.29,0.28,0.38,0.28,0.28,0.36,0.34,0.40,0.36,0.33,0.38])
if switch == 1:
    hasebe = np.array([0.36,0.29,0.28,0.38,0.28,0.28,0.40,0.33,0.38])
if switch == 0:
    hasebe = np.array([0.36,0.29,0.36,0.38,0.36,0.34,0.40,0.33,0.38])



freq = np.array([40,60,78,50,68,89,119,100,140])*GHz

print("\nIn the LFT freqency...")
INTEGRATE = []
for i in range(len(freq)):
    F = freq[i]
    D = define_D(F)
    WL = c/freq[i]
    BW = st.width(F)
    print("\nFreq = {}GHz, Band_width = {}GHz".format(freq[i]/GHz,BW/GHz))
    print("P_CMB({}GHz) ={:.7}".format(freq[i]/GHz, P_CMB(freq[i])))
    print("P_HWP({}GHz) ={:.7}".format(freq[i]/GHz, P_HWP(freq[i])))
    print("P_APT({}GHz) ={:.7}".format(freq[i]/GHz, P_Ape(freq[i])))
    print("P_20K({}GHz) ={:.7}".format(freq[i]/GHz, P_20K(freq[i])))
    print("P_m1({}GHz)  ={:.7}".format(freq[i]/GHz, P_m1(freq[i])))
    print("P_m2({}GHz)  ={:.7}".format(freq[i]/GHz, P_m2(freq[i])))
    print("P_2KF({}GHz) ={:.7}".format(freq[i]/GHz, P_2KF(freq[i])))
    print("P_Lens({}GHz)={:.7}".format(freq[i]/GHz, P_Lens(freq[i])))
    print("P_Det({}GHz)         ={:.7}".format(freq[i]/GHz,P_Det(freq[i])))
    print("P_Opt({}GHz) ={:.7}".format(freq[i]/GHz, P_Opt(freq[i])))
    print("Integratede_P_CMB={:.7}".format(integrate.quad(P_CMB, F-BW, F+BW)[0]/p))
    """
    freq 40.0 , 
    p_cmb  7.44698507836e-24 ,
    p_hwp 2.13635191675e-24 , 
    p_apt 1.6458882942e-23 , 
    p_20K 2.09293683272e-24 , 
    p_m1 1.76179775095e-26 , 
    p_m2 1.76269440977e-26 , 
    p_filter 1.53348853609e-26 , 
    p_lens 1.92529436274e-24 ,
    p_opt 3.01110309396e-23
    """
    #print("int_P_Opt={0:.6}".format(value[0]/p))
    INTEGRATE.append(integrate.quad(P_Opt, F-BW, F+BW)[0])
    print("P_Opt={:.2}, D={}".format(INTEGRATE[i]/p, D))
INTEGEATE = np.array(INTEGRATE)


#hasebe_D1 = np.array([0.36,0.29,0.28,0.38,0.28,])
#delta = hasebe - INTEGEATE
plt.figure()
plt.title("P_opt")
plt.xlabel("Freqency[GHz]")
plt.ylabel("P_opt[pW]")
plt.plot(freq/GHz,hasebe,"o",label="Hasebe-san")
plt.plot(freq/GHz,INTEGEATE/p,"o",label="Takase")
plt.grid()
plt.legend()
"""
plt.figure()
plt.plot(freq/GHz,delta,"o", label="Hase-Taka")
plt.grid()
plt.legend()
"""
plt.show()
