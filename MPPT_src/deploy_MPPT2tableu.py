from loguru import logger as log
import tabpy_client
import pickle

connection = 'http://localhost:9004/'
endemisitas_class = ["Eliminasi", "Rendah", "Sedang", "Tinggi"]

def prediksi_endemisitas(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11, _arg12, _arg13, _arg14, _arg15):
      import numpy as np
      from sklearn.preprocessing import MinMaxScaler
      import pickle
      
      with open("D:\MPPT\model\ml_cls\DTc.pkl", 'rb') as file:
            model = pickle.load(file)
      
      scaler = MinMaxScaler()
      data_test = np.array([_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11, _arg12, _arg13, _arg14, _arg15]).reshape(1,-1)
      data_test = scaler.fit_transform(data_test.reshape(data_test.shape[-1], -1)).reshape(1,-1)
      result = model.predict(data_test)[0]
      
      return endemisitas_class[result]

def prediksi_api( _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11, _arg12, _arg13, _arg14, _arg15):
      import numpy as np
      from sklearn.preprocessing import MinMaxScaler
      import pickle
      
      with open("D:\MPPT\model\ml_reg\RFr.pkl", 'rb') as file:
            model = pickle.load(file)
      
      scaler = MinMaxScaler()
      data_test = np.array([_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9, _arg10, _arg11, _arg12, _arg13, _arg14, _arg15]).reshape(1,-1)
      data_test = scaler.fit_transform(data_test.reshape(data_test.shape[-1], -1)).reshape(1,-1)
      result = model.predict(data_test)[0]
      
      return str(result)
      
def deploy(model_type, connection):
      log.info('model deploy to tableau')
      client = tabpy_client.Client(connection) 
      
      if model_type == 'END':
            client.deploy(
                  'prediksi_endemisitas',
                   prediksi_endemisitas,
                  'mengembalikan status endemisitas', 
                  override = True
            )
            
      if model_type == 'API':
            client.deploy(
                  'prediksi_api',
                   prediksi_api,
                  'mengembalikan nilai prediksi Annual Peracite Incidance', 
                  override = True
            )

deploy('END', connection)
deploy('API', connection)
