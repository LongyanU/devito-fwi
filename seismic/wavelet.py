import numpy as np

def Ricker(t0, tn, dt, f, a=1):
	nt = int(np.ceil((tn - t0 + dt)/dt))
	t = np.linspace(start=t0, stop=tn, num=nt)
	s = t0 or 1./f
	r = np.pi * f * (t - s)
	return a*(1 - 2.*r**2) * np.exp(-r**2)

def Gabor(t0, tn, dt, f, a=1):
	agauss = .5 * f
	tcut = t0 or 1.5/agauss
	nt = int(np.ceil((tn - t0 + dt)/dt))
	t = np.linspace(start=t0, stop=tn, num=nt)	
	s = (t - tcut) * agauss
	return a * np.exp(-2*s**2) * np.cos(2*np.pi*s)

def DGauss(t0, tn, dt, f, a=1):
	nt = int(np.ceil((tn - t0 + dt)/dt))
	t = np.linspace(start=t0, stop=tn, num=nt)
	s = t0 or 1./f
	a = a * (np.pi * f)**2
	r = t - s
	return -2*a*r * np.exp(-a*r**2)