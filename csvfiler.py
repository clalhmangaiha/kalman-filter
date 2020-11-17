import pandas as pd
import numpy as npp

data =pd.read_csv('files2.csv', header=None)
# data.head(1)
# df=pd.DataFrame(data)
data= data.to_numpy()
x = data[:,:1]
y=data[:,1:2]
print (x)
print (y)

