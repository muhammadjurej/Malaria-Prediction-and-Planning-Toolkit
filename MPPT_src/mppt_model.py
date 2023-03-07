import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
from keras_pickle_wrapper import KerasPickleWrapper
import pickle
from loguru import logger as loguru_logger
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MPPT_Model:
      '''
      Terdiri dari beberapa model yang digunakan untuk membuat model prediktif malaria
      model ini mampu memprediksi Annual Parasite Incidence & Status Endemisitas malaria
      
      >>> `untuk ingin memprediksi Annual Parasite Incidence, masukkan predict_model='API'`
      >>> `untuk ingin memprediksi AStatus Endemisitas malaria, masukkan predict_model='END'`
      
      >>> API = Annual Parasite Incidence
      >>> END = Status Endemisitas (0=Eliminasi, 1=Rendah, 2=Sedang, 3=Tinggi)
      '''
      def __init__(self, predict_model:str):
            self.predict_model = predict_model
            self.endemisitas_class = ["Eliminasi", "Rendah", "Sedang", "Tinggi"]
            self.cls_ML = {
                  'KNc':KNeighborsClassifier(),
                  'GNBc':GaussianNB(),
                  'BNBc':BernoulliNB(),
                  'DTc':DecisionTreeClassifier(),
                  'RFc':RandomForestClassifier(),
                  'ABc':AdaBoostClassifier(),
                  'SVc':SVC(),
                  'GBc':GradientBoostingClassifier()
            }
            self.reg_ML = {
                  'RDr':Ridge(),
                  'LSr':Lasso(),
                  'KNr':KNeighborsRegressor(),
                  'DTr':DecisionTreeRegressor(),
                  'RFr':RandomForestRegressor(),
                  'GBr':GradientBoostingRegressor(),
                  'SVr':SVR(),
                  'ABr':AdaBoostRegressor(),
                  'LRr' :LinearRegression()
            }
            
            self.cls_train_acc = {}
            self.cls_test_acc = {}
            self.cls_cfm = {}
            self.reg_train_acc = {}
            self.reg_test_acc = {}
                  
      def input_data(self, X_train, X_test, y_train, y_test):
            '''
            `input_data(X_train, X_test, y_train, y_test, predict_model='API/END)`
            ini wajib masukkan sebelum menjalankan program
            '''
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            if self.predict_model == 'END':
                  self.y_train_onehot = np.zeros((len(y_train), 4))
                  self.y_train_onehot[np.arange((len(y_train))), y_train] = 1

                  self.y_test_onehot = np.zeros((len(y_test), 4))
                  self.y_test_onehot[np.arange((len(y_test))), y_test] = 1
            
                  
      def build_mppt_mlp(self, *hidden):
            '''
            `mppt_mlp([64, 32, 16])`
            untuk membuat model MLP malaria,
            ''' 
            self.mlp_model = keras.Sequential()

            for i, unit in enumerate(hidden):
                  if i == 0:                             
                  #input
                        self.mlp_model.add(keras.layers.Dense(unit, activation='relu', input_shape=(self.X_train.shape[-1],)))
                        keras.layers.Dropout(0.01),
                  #hidden
                  else:
                        self.mlp_model.add(keras.layers.Dense(unit, activation='relu'))
            #output
            if self.predict_model == 'END':
                  self.mlp_model.add(keras.layers.Dense(4, activation='softmax'))
            
            elif self.predict_model == 'API':
                  self.mlp_model.add(keras.layers.Dense(1))
                  
      def mppt_mlp_summary(self):
            '''
            `mppt_mlp_summary()`
            menampilkan rangkuman model mlp
            '''
            self.mlp_model.summary()
            
      def mppt_mlp_fit(self, batch_size=8, epoch=100):
            '''
            `mppt_mlp_fit(batch_size=8, epoch=100)`
            melatih model mlp mppt
            '''
            
            if self.predict_model == 'END':
                  checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        filepath='model\mlp_cls\ckpt/end_model.h5', 
                        monitor='val_loss', verbose=0,
                        save_best_only=True, mode='min')
                  
                  self.mlp_model.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-2), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])
                  
                  kpw = KerasPickleWrapper(self.mlp_model)
                  kpw().fit(
                        self.X_train,
                        self.y_train_onehot,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=1,
                        callbacks=checkpoint,
                        validation_split = 0.2,
                  )
                  kpw().load_weights('model\mlp_cls\ckpt/end_model.h5')
                  kpw().save('model\mlp_cls/end_model.h5')
                  
            elif self.predict_model == 'API':
                  checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        filepath='model\mlp_reg\ckpt/api_model.h5',
                        save_weights_only=True,
                        monitor='val_loss',
                        save_best_only=True, 
                        mode='min')
                  
                  self.mlp_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                                          loss='mean_squared_error',
                                          metrics=['mae', 'mse']
                                          )
                  kpw = KerasPickleWrapper(self.mlp_model)
                  kpw().fit(
                        self.X_train,
                        self.y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=1,
                        callbacks=checkpoint,
                        validation_split = 0.2,
                  )
                  
                  kpw().load_weights('model\mlp_reg\ckpt/api_model.h5')
                  kpw().save('model\mlp_reg/api_model.h5')
                  
      def mppt_mlp_evaluate(self):
            '''
            `mppt_mlp_evaluate()`
            mengevaluali model mppt mlp, dengan melihat loss, accuracy, dan mae
            '''
            #self.mlp_model = keras.models.load_model(self.mlp_model_name)
            
            if self.predict_model == 'END':
                  loss, accuracy =  self.mlp_model.evaluate(self.X_test, self.y_test_onehot)
                  loguru_logger.info(f'Loss: {loss} & Accuracy: {accuracy}')
            elif self.predict_model == 'API':
                  loss, mae, mse = self.mlp_model.evaluate(self.X_test, self.y_test)
                  loguru_logger.info(f'loss: {loss}, mae: {mae}, mse: {mse}')
                  
      def load_mppt_mlp(self, model_path):
            '''
            `load_mppt_model(model_path)
            '''
            loguru_logger.info(f"model mlp mppt {model_path} loaded!")
            return load_model(model_path)
                  
      def mppt_mlp_predict(self, data, model, scale_data=False):
            '''
            `mppt_model_predict(data)`
            fungsi ini hanya bisa digunaakan dengan model yang diload
            load_mppt_mlp_model(path)
            memprediksi hasil API/END berdasarkan data baru
            '''
            flatten = keras.layers.Flatten()
            data = tf.reshape(flatten(data), (1, 15))
            
            if scale_data == True:
                  minmaxScaler = MinMaxScaler() # tanpa nilai minus
                  data = minmaxScaler.fit_transform(data)
            
            if self.predict_model == 'END':
                  class_idx = np.argmax(model.predict(data), axis=1)
                  out = "status: " + self.endemisitas_class[class_idx[0]]
                  loguru_logger.info(f'Hasil prediksi Endimisitas Malaria -> {self.endemisitas_class[class_idx[0]]}')
            
            elif self.predict_model == 'API':
                  out = model.predict(data)
                  loguru_logger.info(f'Hasil prediksi API Malaria -> {out}')
            
            return out
      
      def train_mppt_ml(self):
            '''
            `train_ml()`
            melakukan training untuk semua model ML untuk klasifikasi dan regresi
            >>> klasifikasi untuk prediksi ENDEMISITAS malria
            >>> Regresi untuk prediksi Annual Parasite Incidence
            Berikut model yang digunakan
            
            Klasifikasi
            -----------
            >>> K-NeighborsClassifier
            >>> Gaussian_NB
            >>> Bernouli_NB
            >>> DecisionTreeClassifier
            >>> RandomForestClassifier
            >>> AdaBosstClassifier
            >>> SVC
            >>> GradientBoostingClassifier
            
            Regresi
            -------
            >>> Ridge
            >>> Lasso
            >>> K-NeighborsRegresor
            >>> DecisionTreeRegressor
            >>> RandomForestRegressor
            >>> GradientBoostingRegressor
            >>> SVR
            >>> AdaboostRegressor
            '''
            if self.predict_model == 'END':
                  loguru_logger.info("Training Model Machine Learning Untuk Klasifikasi ENDEMISITAS MLARIA")
                  for name, model in self.cls_ML.items():
                        model.fit(self.X_train, self.y_train)
                        y_pred_train = model.predict(self.X_train)
                        y_pred_test = model.predict(self.X_test)
                        acc_train = accuracy_score(self.y_train, y_pred_train)
                        acc_test = accuracy_score(self.y_test, y_pred_test)
                        cfm = confusion_matrix(self.y_test, y_pred_test)
                        self.cls_train_acc[name] = acc_train
                        self.cls_test_acc[name] = acc_test
                        self.cls_cfm[name] = cfm
                        
            elif self.predict_model == 'API':
                  loguru_logger.info("Training Model Machine Learning Untuk Regresi Annual Parasite Incidence(API)")
                  for name, model in self.reg_ML.items():
                        model.fit(self.X_train, self.y_train)
                        y_pred_train = model.predict(self.X_train)
                        y_pred_test = model.predict(self.X_test)
                        acc_train = r2_score(self.y_train, y_pred_train)
                        acc_test = r2_score(self.y_test, y_pred_test)
                        self.reg_train_acc[name] = acc_train
                        self.reg_test_acc[name] = acc_test
                             
      def find_and_save_best_ml_model(self, save_path):
            '''
            `find_and_save_best_ml_model(savepath='')`
            save best ML model untuk ENDEMISITAS MALARIA
            '''
            best_model = ''
            best_acc = 0
            if self.predict_model == 'END':
                  assert self.cls_train_acc != {}, "Model ML belum dilatih!, jalankan MPPT_Model.ml() dulu"
                  for model, acc in self.cls_test_acc.items():
                        if acc > best_acc:
                              best_acc = acc
                              best_model = model
                  loguru_logger.info(f'Best Model ENDEMISITAS MALARIA -> {best_model} dengan akurasi {best_acc * 100}%')
                  model_name = save_path + f'/{best_model}.pkl'
                  
                  model_choice = self.cls_ML[best_model].fit(self.X_train, self.y_train)
                  with open(model_name, 'wb') as file:
                        pickle.dump(model_choice, file)
                  loguru_logger.info(f'model saved {model_name}')
                  
            
            elif self.predict_model == 'API':
                  assert self.reg_train_acc != {}, "Model ML belum dilatih!, jalankan MPPT_Model.ml() dulu"
                  for model, acc in self.reg_test_acc.items():
                        if acc > best_acc:
                              best_acc = acc
                              best_model = model
                  loguru_logger.info(f'Best Model Annual Parasite Incidence(API) -> {best_model} dengan akurasi {best_acc * 100}%')
                  model_name = save_path + f'/{best_model}.pkl'
                  
                  model_choice = self.reg_ML[best_model].fit(self.X_train, self.y_train)
                  with open(model_name, 'wb') as file:
                        pickle.dump(model_choice, file)
                  loguru_logger.info(f'model saved {model_name}')
                  
      def all_ml_acc_and_cfm(self):
            '''
            `all_ml_acc_and_cfm()` -> train_acc, test_acc, cfm
            Kembalikan nilai akurasi & confution matrix untuk semua algoritma ML
            '''
            if self.predict_model == 'END':
                  return self.cls_train_acc, self.cls_test_acc, self.cls_cfm
            elif self.predict_model == 'API':
                  return self.reg_train_acc, self.reg_test_acc, 0
            
      def load_mppt_ml(self, model_path):
            '''
            `load_ml_best_model(model_path)`
            load best model ml
            '''
            with open(model_path, 'rb') as file:
                  best_model = pickle.load(file)
                  
            loguru_logger.info(f"ml best model {model_path} loaded!")
            return best_model
      
      def mppt_ml_predict(self, data_test, model, scale_data=False):
            '''
            `best_ml_predict(data, model)`
            predic API/END dengan ML
            '''
            if scale_data == True:
                  minmaxScaler = MinMaxScaler() # tanpa nilai minus
                  data_test = minmaxScaler.fit_transform([data_test])[0]
            
            if self.predict_model == "END":
                  class_idx = model.predict([data_test])[0]
                  out = "status: " + self.endemisitas_class[class_idx]
                  loguru_logger.info(f'Hasil prediksi Endimisitas Malaria -> {self.endemisitas_class[class_idx]}')
                  
            elif self.predict_model == "API":
                  out = model.predict([data_test])[0]
                  loguru_logger.info(f'Hasil prediksi API Malaria -> {out}')
            
            return out
                        
      def viz_confusion_matrix(self, model_name, cfm):
            '''
            `viz_confusion_matrix(model_name="RFc", cfm)`
            mencetak confusion matrix
            Status Endemisitas (0=Eliminasi, 1=Rendah, 2=Sedang, 3=Tinggi)
            '''
            assert self.predict_model == 'END', 'Regresi Tidak dihitung Confusion matrix '
            fig, ax = plt.subplots()
            im = ax.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # membuat label pada sumbu x dan y
            classes = ['Eliminasi', 'Rendah', 'Sedang', 'Tinggi']
            ax.set(xticks=np.arange(cfm.shape[1]),
                  yticks=np.arange(cfm.shape[0]),
                  xticklabels=classes, yticklabels=classes,
                  title=f'Confusion Matrix {model_name}',
                  ylabel='True label',
                  xlabel='Predicted label',
            )
            # menampilkan nilai pada setiap cell
            for i in range(cfm.shape[0]):
                  for j in range(cfm.shape[1]):
                        ax.text(j, i, format(cfm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cfm[i, j] > cfm.max() / 2. else "black")

            # membuat layout plot agar rapih
            fig.tight_layout()
            plt.show()     
                              
                  
                              
                        