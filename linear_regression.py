import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Lineardata_train.csv')
tr=np.array(df)
dft=pd.read_csv('Lineardata_test.csv')
ts=np.array(dft)
x=tr[1:]
y=tr[0]
r=ts[0]
rx=ts[1:]
def lr(x,y,rx,r):
  m,i=np.polyfit(x,y,deg=1)
  p=np.add(np.multiply(rx,m),i)
  diff=np.subtract(p,r)
  sumofsqa=0
  for i in diff:
    sumofsqa=sumofsqa+(i*i)
  cost=sumofsqa/(2*len(diff))
  return(cost,m,i)
costx=[]
iteration=[]
I=0
for i in range(0,20):
  cost,m,i=lr(x[i],y,rx[i],r)
  costx.append(cost)
  I=I+1
  iteration.append(I)
plt.plot(iteration,costx)
print(min(costx))
