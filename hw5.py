import numpy as np
import math
import matplotlib.pyplot as plt

data = np.load('data.npy')

#訊號產生器
def F(t,A,B,C,tc,beta,w,phi):
    return A+B*((tc-t)**beta)*(1+C*np.cos(w*np.log(tc-t)+phi))
#energy func.
def E(t,b,A,B,C,tc,beta,w,phi):
    return np.sum(abs(b-F(t,A,B,C,tc,beta,w,phi))) 

t = np.arange(1166) #t = 0~tc
b = np.log(data[:1166])

#先預設Ａ,B,C的值
A = 1 #A>0
B = -1 #B<0
C = 0.5 #C is close to zero,C!=0

pop = np.random.randint(0,2,(10000,34)) #初代人口
fit = np.zeros((10000,1))

X = np.zeros((1166,3))

for i in range(10):
    for generation in range(10):
        print(generation)
        for i in range(10000):  
            gene = pop[i,:]
            tc = np.sum(2**np.array(range(4))*gene[:4])+1151  #tc is in [1151,1166]
            beta = np.sum(2**np.array(range(10))*gene[4:14])/1000
            w = (np.sum(2**np.array(range(10))*gene[14:24]))/100
            phi = np.sum(2**np.array(range(10))*gene[24:])/100
        sortf = np.argsort(fit[:,0])
        pop = pop[sortf,:]  
        for i in range(100,10000): #把活得最好的100個人交配生成另外9900個人取代原本的
            fid = np.random.randint(0,100)  #父，取0-99隨便一個數字做id
            mid = np.random.randint(0,100)  #母
            while mid == fid:  
                mid = np.random.randint(0,100) #避免父母取到同個人
            mask = np.random.randint(0,2,(1,34))
            son = pop[mid,:]  
            father = pop[fid,:]
            son[mask[0,:]==1] = father[mask[0,:]==1]  
            pop[i,:] = son
        for i in range(1000): #突變
            m = np.random.randint(0,10000)
            n = np.random.randint(0,34)
            pop[m, n] = 1-pop[m,n] #第m人第n個基因1變0，0變1
            
    for i in range(tc):
        X[i,0] = 1
        X[i,1] = ((tc-t[i])**beta)
        X[i,2] = np.cos(w*np.log(tc-t[i])+phi)*((tc-t[i])**beta)
    x = np.linalg.lstsq(X, b)[0] #least square
    A = x[0]
    B = x[1]
    C = x[2]/x[1] #因為原式展開後會多乘Ｂ  

signals = []
end = 0
for i in range(1166):
    answer = A + B*((tc - i)**beta)*(1 + C*np.cos(w*np.log(tc - i) + phi))
    answer = 2.71828**answer
    if math.isnan(answer): 
        end = i
        break
    signals.append(answer)
    
plt.plot(data[:end],label='data')
plt.plot(signals,label='signal')
plt.legend(loc='best')
plt.show()
