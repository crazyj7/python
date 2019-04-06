
import numpy as np
import pandas as pd



# df = pd.DataFrame({'a':[np.nan,1,2,np.nan,np.nan,5,np.nan,np.nan]}, index=[0,1,2,3,4,5,6,7])
df = pd.DataFrame({'a':[np.nan,1,2,np.nan,np.nan,5,6,np.nan]}, index=[0,1,2,3,4,5,6,7])
print(df)
r=df.interpolate(method='index', limit=1, limit_direction='backward')
print(r)


valargmax=np.max(np.where((df.isnull().eq(False).values==True).flatten()==True))
print(valargmax)

r = df[0:(valargmax+1)].interpolate(method='index').append(df[(valargmax+1):])
print(r)






