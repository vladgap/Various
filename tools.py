import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
from MLNN import *

tools_version = '2.1'
print(f' Version of tools is {tools_version}\n',
      'class NN2to1 -- without external scaler')

def CopyPasteToPandas(a):
    if a.startswith('\n'):
        a = a[1:]
    if a.endswith('\n'):
        a = a[:-1]
    rows = a.split('\n')
    data = []
    for row in rows:
        parts = row.split('\t')
        parsed_row = []
        for item in parts:
            item = item.strip()
            try:
                # נסה להמיר ל-int או float
                if '.' in item:
                    parsed_row.append(float(item))
                else:
                    parsed_row.append(int(item))
            except ValueError:
                parsed_row.append(item)
        data.append(parsed_row)
    return data

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


class NN2to1:
  def __init__(self, X, T, mesh, confidences=None, hidden_layers=1, hidden_activation='linear'):
    self.X=X
    self.T=T
    self.mesh=mesh
    self.confidences=confidences
    self.hidden_layers=hidden_layers
    self.hidden_activation=hidden_activation
    self.network=NN(layers=[2,hidden_layers,1], hidden_activation = hidden_activation)
    pd.options.plotting.backend = "plotly"

  def fit_model(self, epochs=1000,  learning_rate = 0.001, momentum_term = 0.95):
    self.network.fit(self.X, self.T, epochs=epochs, confidences=self.confidences, learning_rate = learning_rate, momentum_term = momentum_term)
    print('Initial loss =', self.network.loss_list[0])
    print('Final loss =', self.network.loss_list[-1])
    fig=pd.Series(self.network.loss_list).plot()
    fig.show()

  def import_weights(self,weights):
    self.network.import_weights(weights)

  def export_weights(self):
    print ('Hidden layers:', self.hidden_layers)
    print ('Hidden activation:', self.hidden_activation)
    print ('Loss:', self.network.loss_list[-1],'\n')
    return self.network.export_weights()

  def print_weights(self):
    self.network.print_weights()

  def show(self):
    self.predics=self.network.predict(self.X)
    self.errors=(self.predics[:,0]-self.T[:,0])/self.T[:,0]*100
    self.mesh_predics=self.network.predict(self.mesh)
    self.__plot()

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
    fig.update_scenes(xaxis_title='x', yaxis_title='y',
                      camera_eye=dict(x=0, y=-2.2, z=0),
                      aspectratio=dict(x=1, y=1, z=1)
                      )
    fig.update_scenes(camera_projection_type="orthographic")
    self.fig = fig
    fig.show()



def print_scaler_data(scaler_data):
    print('scaler X -- mean, stdev:  ',scaler_data[0], scaler_data[1])
    print('scaler T -- mean, stdev:  ',scaler_data[2], scaler_data[3])


class Fit2to1:
  def __init__(self, X, T, mesh, confidences=None, hidden_layers=1, hidden_activation='linear'):
    self.X=X
    self.T=T
    self.mesh=mesh
    self.confidences=confidences
    self.hidden_layers=hidden_layers
    self.hidden_activation=hidden_activation
    self.scaler_X, self.scaler_T = self.__get_scalers()
    self.X_sc = self.scaler_X.transform(X)
    self.T_sc = self.scaler_T.transform(T)
    self.network=VectorBackProp(layers=[2,hidden_layers,1], hidden_activation = hidden_activation)
    pd.options.plotting.backend = "plotly"

  def fit_model(self, epochs=1000,  learning_rate = 0.001, momentum_term = 0.95):
    self.network.fit(self.X_sc, self.T_sc, epochs=epochs, confidences=self.confidences, learning_rate = learning_rate, momentum_term = momentum_term)
    print('Initial loss =', self.network.loss_list[0])
    print('Final loss =', self.network.loss_list[-1])
    fig=pd.Series(self.network.loss_list).plot()
    fig.show()

  def import_weights(self,weights):
    self.network.import_weights(weights)

  def export_weights(self):
    print ('Hidden layers:', self.hidden_layers)
    print ('Hidden activation:', self.hidden_activation)
    print ('Loss:', self.network.loss_list[-1],'\n')
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
                      camera_eye=dict(x=0, y=-2.2, z=0),
                      aspectratio=dict(x=1, y=1, z=1)
                      )
    fig.update_scenes(camera_projection_type="orthographic")
    self.fig = fig

    fig.show()


def export_weights_as_pandas(bp):
    a=bp.export_weights()
    b=pd.DataFrame()
    for i in a:
        b=pd.concat([b,pd.DataFrame(i)], axis=0)
    return b

def round_to_significant_digits(num, keep_int=False):
    if keep_int and isinstance(num, int):
        return num
    if not isinstance(num, (int, float)):
        return num  # Return non-numeric types as is
    abs_num = abs(num)
    if abs_num >= 100:
        return round(num)    # Existing rules for numbers less than 100
    elif abs_num >= 0.1:
        return float(f'{num:.3g}')        # 3 significant digits
    elif 0.01 <= abs_num < 0.1:
        return float(f'{num:.2g}')        # 2 significant digits
    else: # abs_num < 0.01
        return float(f'{num:.1g}')        # 1 significant digit

def apply_rounding_to_structure(data_structure, rounding_func, **kwargs):
    if isinstance(data_structure, pd.DataFrame):
        # For DataFrames, map is used, and kwargs can be passed directly
        return data_structure.map(lambda x: rounding_func(x, **kwargs))
    elif isinstance(data_structure, list):
        rounded_list_of_lists = []
        for inner_list in data_structure:
            rounded_inner_list = []
            for item in inner_list:
                rounded_inner_list.append(rounding_func(item, **kwargs))
            rounded_list_of_lists.append(rounded_inner_list)
        return rounded_list_of_lists
    else:
        raise TypeError("Input must be a pandas DataFrame or a list of lists")



