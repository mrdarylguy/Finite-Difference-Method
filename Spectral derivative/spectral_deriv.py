import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({"font.size": 18})

n=64
L=30
dx=L/n
x=np.arange(-L/2, L/2, dx, dtype="complex_")

#create function
f = np.cos(x)*np.exp(-np.power(x, 2)/25)

#analytic derivaive
df = -(np.sin(x) * np.exp(-np.power(x, 2)/25 + (2/25)*x*f))

#FDM
#create array
dfFD = np.zeros(len(df), dtype='complex_')
#Iterate across array
for kappa in range(len(df)-1):
    dfFD[kappa] = (f[kappa+1] - f[kappa])/dx

dfFD[-1] = dfFD[-2]

#Spectral derivative
#Approximate using FFT
fhat = np.fft.ifft(f)
kappa = (2*np.pi/L)*np.arange(-n/2, n/2)

#re-order frequencies
kappa = np.fft.fftshift(kappa)

#obtain real part of function for plotting
dfhat = kappa*fhat*(1j)

#Inverse fourier transform
dfFFT = np.real(np.fft.ifft(dfhat))

#plot results
plt.plot(x, df.real, color="k", LineWidth=2, label="true derivative")
plt.plot(x, dfFD.real, "--", color="b", LineWidth=1.5, label="finite difference")
plt.plot(x, dfFFT.real, "--", color="c", LineWidth=1.5, label='spectral dervative')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()


