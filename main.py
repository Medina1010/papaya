import numpy as np
import matplotlib.pyplot as plt

V = np.loadtxt('FormaV.txt')
T = np.loadtxt('FormaT.txt')

print('macimo',np.max(V[0:,1]))
print('mijimo',np.min(V[0:,1]))

largoi = 0.09974571654926101
anchoi = 0.058115436648410505

largo = largoi
ancho = anchoi

V[0:,1] = V[0:,1] * ((largo/largoi*np.cos(V[0:,0]))**2+(ancho/anchoi*np.sin(V[0:,0]))**2)**(1/2)

X = np.linspace(-np.pi, np.pi, 100)

Ve = np.zeros_like(X)
e = np.zeros_like(X)
for v in V:
    Ve[int(50/np.pi*(v[0]+np.pi))] += v[1]
    e[int(50/np.pi*(v[0]+np.pi))] += 1
for n in range(0,100):
    if e[n] ==0:
     #   print(n)
        Ve[n] = Ve[n+1]
        e[n] +=1

Ve[30]=Ve[31]
Ve[76]=Ve[77]

Ve/=e

Ve = Ve[50:]

Te = np.zeros_like(X)
e = np.zeros_like(X)
for v in T:
    Te[int(50/np.pi*(v[0]+np.pi))] += v[1]
    e[int(50/np.pi*(v[0]+np.pi))] += 1
for n in range(0,100):
    if e[n] ==0:
        #print(n)
        Te[n] = Te[n+1]
        e[n] +=1
Te/=e

for n in range(0,100):
    if Te[n] >600:
        print(n)

Te[45] = Te[46]
Te[13] = Te[14]
Te[58] = Te[59]
Te[19] = Te[20]
Te[60] = Te[61]
Te[71] = Te[70]
Te[72] = Te[70]
Te[77] = Te[76]
Te[78] = Te[76]
Te[79] = Te[80]
Te[81] = Te[80]
Te[82] = Te[84]
Te[83] = Te[84]

#plt.plot(V[0:,0],V[0:,1])
#plt.plot(V[50,0],V[50,1],'.')
#plt.plot(T[0:,1])
fou = np.fft.fft(Te)

s = np.zeros_like(X)
for i in range(0,4):
    s += fou[i*5].real*np.cos(5*i*(X+np.pi))/100
plt.plot(X, Te)
plt.plot(X, s)
plt.savefig('test1.png')
plt.cla()

print('masimo', np.max(s))
print('micimo', np.min(s))


#fou = np.fft.fft(Ve)
#s = np.zeros_like(X[50:])
#for i in range(0,40):
#    s += fou[i].real*np.cos(2*i*(X[50:]))/50
plt.plot(X[50:], Ve)
plt.plot(X[50:], np.poly1d(np.polyfit(X[50:], Ve,14))(X[50:]))
#plt.plot(X[50:], s)
plt.savefig('test2.png')



def rtheta (theta):
    s = 0
    for i in range(0,4):
        s += fou[i*5].real*np.cos(5*i*(theta+np.pi))/100
    return s/385

rphi = np.poly1d(np.polyfit(X[50:], Ve,14))

import scipy as spy
from scipy import integrate

def f(phi, theta):
    return (rtheta(theta)*rphi(phi))**2*np.sin(phi)

result, stderr = integrate.dblquad(f,0,2*np.pi, 0, np.pi)

print(result)

result, stderr = integrate.dblquad(f,0,2*np.pi, 0, np.pi)

def f3(phi, theta):
    return (rtheta(theta)*rphi(phi))**3*np.sin(phi)/3

result, stderr = integrate.dblquad(f3,0,2*np.pi, 0, np.pi)

print(result*1000000, 1052/(result*1000000))

print(0.7950280140235211*result*1000000)
