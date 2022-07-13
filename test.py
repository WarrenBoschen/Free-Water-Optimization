import plotly.graph_objects as go
import numpy as np

x = np.arange(0.1,1.1,0.1)
y = np.linspace(-np.pi,np.pi,10)
#print(x)
#print(y)

X,Y = np.meshgrid(x,y)
#print(X)
#print(Y)
result = []

for i,j in zip(X,Y):
    result.append(np.log(i)+np.sin(j)) 

result[0][0] = float("nan")

upper_bound = np.array(result)+1
lower_bound = np.array(result)-1

fig = go.Figure(data=[
    go.Surface(z=result),
    go.Surface(z=upper_bound, showscale=False, opacity=0.3,colorscale='purp'),
    go.Surface(z=lower_bound, showscale=False, opacity=0.3,colorscale='purp')])
fig.show()