# <center>Malaria-Prediction-and-Planning-Toolkit<center>
This is a toolkit that is used to help predict the number of malaria incidents in an area, and is able to see the status of malaria endemicity :rocket:

These tools train multiple models to predict Annual Paracite Incidence and Endemicity Status.

### :star: For Endemicity Status I use the model :
 - K-NeighborsClassifier
 - Gaussian_NB
 - Bernouli_NB
 - DecisionTreeClassifier
 - RandomForestClassifier
 - AdaBosstClassifier
 - SVC
 - GradientBoostingClassifier
 - MLP [128, 32, 16, 8]
 
 After Training All model for Endemysitey, i found DecisionTreeClassifier is the best model to predict **status of malaria endemicity** with accuracy 92.9 % :smile:
 
 ### :star: For predict the number of malaria incidents in an area I use the model :
 - Ridge
 - Lasso
 - K-NeighborsRegresor
 - DecisionTreeRegressor
 - RandomForestRegressor
 - GradientBoostingRegressor
 - SVR
 - AdaboostRegressor
 - MLP Regressor [128, 32, 16, 8]
 
 After Training All model for Malaria Incidence, i found RandomForestRegressor is the best model to predict **Number of malara incidence** with accuracy 93.91 % :smile:
 
 ## :tv: DASHBOARD Malaria Prediction and Planning Toolkit 
 
 To improve the performance of this system, I added a dashboard that can see trends in malaria development with several important indicators, such as population density, climate change, rainfall, and the number of hospitals. And with this dashboard too, users can predict endemicity status and number of malaria incidence, by entering several important parameters such as:
 
 - the size of an area
- population density (people/kilometers squared)
- percentage of access to sanitation
- rainfall
- Number of health centers
- number of specialized hospitals
- number of regional public hospitals
- poverty percentage
- the number of residents with insurance
- minimum temperature
- maximum temperature
- average temperature
- total population
 
 
before using the dashboard, we have to install tabpy 
 ```
 pip install tabpy
 ```
 
 and then run this command in your terminal
 
 ```
 tabpy
 ```

and below is the display of the dashboard

![dashboard](https://user-images.githubusercontent.com/46664825/223641186-be6e7e6e-bba9-40b8-83b5-29135349373e.JPG)

 
 
