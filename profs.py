import cv2
import numpy as np
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'],
                   'Age': [25, 30, 35],
                   'City': ['New York', 'Los Angeles', 'Chicago']})


# Extract a subset of rows using a boolean condition
df2 = df.loc[:,['Name','Age']]
df3 = df2.loc[df2['Name']!='Bob']
idx = df3['Age'].idxmax()
df4 = df3.loc[idx]

#slope1 = lambda yq, yc, xq, xc: (((yq-yc)/(xq-xc))-1.9822)/((1.9822*((yq-yc)/(xq-xc)))+1)
slope1 = lambda yq, yc, xq, xc: (((yq-yc)/(xq-xc))+3.077)/((-3.077*((yq-yc)/(xq-xc)))+1)
#slope2 = lambda mc: (mc+3.38)/((-3.38*mc)+1)
slope2 = lambda mc: (mc+1)/((1*mc)+1)
print(slope2(slope1(0,4,4,0)))