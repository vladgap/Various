import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
from MLNN import *

vladgap_tools_version = '0.1'
print(f'Version of vladgap_tools is {vladgap_tools_version}')


def CopyPasteToArray(a):
    if a.startswith('\n'):
        b=a[1:]
    else:
        b=a
    if b.endswith('\n'):
        b=b[:-1]
    c=b.replace('\t',',')
    d=c.split('\n')
    f=[]
    for e in d:
        if e.replace(',','').replace('.','').isdigit(): # only digits no letters
            f.append(list(eval(e)))
        else:
            f.append(e.split(','))
    return f


def print_scaler_data(scaler_data):
    print('scaler X -- mean, stdev:  ',scaler_data[0], scaler_data[1])
    print('scaler T -- mean, stdev:  ',scaler_data[2], scaler_data[3])


class Fit2to1:
    def __init__(self,X,T,mesh,hidden_layers=1,hidden_activation='linear'):
        self.X=X
        self.T=T
        self.mesh=mesh
        self.hidden_layers=hidden_layers
        self.hidden_activation=hidden_activation
        self.scaler_X, self.scaler_T = self.__get_scalers()
        self.X_sc = self.scaler_X.transform(X)
        self.T_sc = self.scaler_T.transform(T)
        self.network=VectorBackProp(layers=[2,hidden_layers,1], hidden_activation = hidden_activation)
        pd.options.plotting.backend = "plotly"

    def fit_model(self, epochs=1000, learning_rate = 0.001, momentum_term = 0.95):
        self.network.fit(self.X_sc, self.T_sc, epochs=epochs, learning_rate = learning_rate, momentum_term = momentum_term)
        print('Initial loss =', self.network.loss_list[0])
        print('Final loss =', self.network.loss_list[-1])
        fig=pd.Series(self.network.loss_list).plot()
        fig.show()

    def import_weights(self,weights):
        self.network.import_weights(weights)

    def export_weights(self):
        # print ('Hidden layers:', self.hidden_layers)
        # print ('Hidden activation:', self.hidden_activation)
        # print ('Loss:', self.network.loss_list[-1],'\n')
        return self.network.export_weights()

    def print_weights(self):
        self.network.print_weights()

    def print_scaler_data(self):
        print('scaler X -- mean, stdev:  ',self.scaler_X.mean_, self.scaler_X.scale_)
        print('scaler T -- mean, stdev:  ',self.scaler_T.mean_, self.scaler_T.scale_)

    def show(self):
        self.predics=self.scaler_T.inverse_transform(self.network.run(self.X_sc))
        self.errors=(self.predics[:,0]-self.T[:,0])/self.T[:,0]*100
        self.mesh_predics=self.scaler_T.inverse_transform(self.network.run(self.scaler_X.transform(self.mesh)))
        self.__plot()

    def __get_scalers(self):
        scaler_X = preprocessing.StandardScaler().fit(self.X)
        scaler_T = preprocessing.StandardScaler().fit(self.T)
        return scaler_X, scaler_T

    def __plot(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Errors','Model'), column_widths=[0.5, 0.5],
                        specs=[[{"secondary_y": True}, {"type": "scene"}]])

        fig.add_trace(go.Scatter(x=self.T[:,0] , y=self.predics[:,0], mode='markers', marker_size=4, name='Predics', marker_color='black' ), 1, 1)
        fig.add_trace(go.Scatter(x=self.T[:,0], y=self.T[:,0], mode='lines', line_color='red', line_width=0.2, showlegend=False),1,1,secondary_y=False)
        fig.add_trace(go.Scatter(x=self.T[:,0] , y=self.errors, mode='markers', marker_size=4, name='Errors', marker_color='orange' ), 1, 1, secondary_y=True,)

        fig.add_trace(go.Scatter3d(x=self.X[:,0], y=self.X[:,1], z=self.T[:,0], mode='markers', name='Data'), 1, 2)
        fig.add_trace(go.Scatter3d(x=self.mesh[:,0], y=self.mesh[:,1], z=self.mesh_predics[:,0], mode='markers',marker_color='green', marker_size=1, name='Mesh'),1,2)

        fig.update_layout(title='', autosize=True,
                        # width=1550,
                        height=500,
                        margin=dict(l=0, r=0, b=0, t=30))
        fig.update_scenes(xaxis_title='X1', yaxis_title='X2',
                        camera_eye=dict(x=0, y=-2.2, z=0)
                        )
        fig.update_scenes(camera_projection_type="orthographic")

        fig.show()


def export_weights_as_pandas(bp):
    a=bp.export_weights()
    b=pd.DataFrame()
    for i in a:
        b=pd.concat([b,pd.DataFrame(i)], axis=0)
    return b