#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def F1(t): #linear的訊號產生器
    return 0.063*(t**3)-5.284*(t**2)+4.887*t+412+np.random.normal(0,1) #最後加的常態是noise
#會遮住四個數字看能不能還原出相近的參數

def F2(t,A,B,C,D): #t是vector
    return A*(t**B)+C*np.cos(D*t)+np.random.normal(0,1,t.shape)

#######hw########
def F3(A,B,C,beta,e,tc,phi):
    signal = np.zeros((1,tc))
    for i in range(tc):
        signal
y = np.load('data.npy')
#energy func.
def E(t,b,A,B,C,D):
    return np.sum(abs(b - F2(t,A,B,C,D)))
    
n = 1000
b = np.zeros((n,1))
t = np.zeros((n,1))
for i in range(n): #產生1000pair的t跟F1(t)
    t[i] = np.random.random()*100
    b[i] = F1(t[i])
    
A = np.zeros((n,5))
for i in range(n):
    A[i,0] = t[i]**4
    A[i,1] = t[i]**3
    A[i,2] = t[i]**2
    A[i,3] = t[i]
    A[i,4] = 1

x = np.linalg.lstsq(A,b)[0] #least square
print(x)

t2 = np.random.random((n,1))*100
b2 = F2(t2,0.6,1.2,100,0.4)

#exhaustive search
print(E(t2,b2,0.6,1.2,100,0.4))
'''
X = np.zeros((1024,1))
Y = np.zeros((1024,1))
Z = np.zeros((1024,1024))
for i in range(1024):
    X[i] = -5.11+i*0.01
    for j in range(1024):
        Y[j] = -511+j
        Z[i,j] = E(t2,b2,X[i],1.2,Y[j],0.4)
plt.plot(X,Y)
'''
 
pop =np.random.randint(0,2,(10000,40)) #初代人口
fit = np.zeros((10000,1))

for generation in range(10):
    for i in range(10000):
        gene = pop[i,:]
        A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100 #-5.11~5.12
        B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
        C = (np.sum(2**np.array(range(10))*gene[20:30])-511)
        D = (np.sum(2**np.array(range(10))*gene[30:])-511)/100
        fit[i] = E(t2,b2,A,B,C,D)
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
    for i in range(100,10000): #把活得最好的100個人交配生成另外9900個人取代原本的
        fid = np.random.randint(0,100) #父，取0-99隨便一個數字做id
        mid = np.random.randint(0,100) #母
        while mid==fid:
            mid = np.random.randint(0,100) #避免父母取到同個人
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:]
        father = pop[fid,:]
        son[mask[0,:]==1] = father[mask[0,:]==1]
        pop[i,:] = son
    for i in range(1000): #突變
        m = np.random.randint(0,10000)
        n = np.random.randint(0,40)
        pop[m,n] = 1-pop[m,n]
    
for i in range(10000):
    gene = pop[i,:]
    A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100 
    B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
    C = (np.sum(2**np.array(range(10))*gene[20:30])-511)
    D = (np.sum(2**np.array(range(10))*gene[30:])-511)/100
    fit[i] = E(t2,b2,A,B,C,D)
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]

gene = pop[0,:] #拿第0個人的ＡＢＣＤ
A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100 
B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
C = (np.sum(2**np.array(range(10))*gene[20:30])-511)
D = (np.sum(2**np.array(range(10))*gene[30:])-511)/100
print('A',A,'B',B,'C',C,'D',D)