import streamlit as s
import pandas as pd 
import random
from sklearn.preprocessing import StandardScaler
import time
all_values = []

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
s.title('California Housing Price Prediction')
s.image('https://img.onmanorama.com/content/dam/mm/en/lifestyle/decor/images/2023/6/1/house-middleclass.jpg')
s.write('This model should learn from the data and be able to predict the median housing price in any district, given all the other metrics.')


s.header('Model of housing prices to predict median house values in California ',divider=True)

s.subheader('''User Must Enter Given values to predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

s.sidebar.title('Select from Select House Features')
s.sidebar.image('https://images.pexels.com/photos/462358/pexels-photo-462358.jpeg?cs=srgb&dl=architectural-design-architecture-blue-sky-462358.jpg&fm=jpg')


temp_df = pd.read_csv('calfornia.csv')

random.seed(12) 
for i in temp_df[col]:
    min_value,max_value = temp_df[i].agg(['min','max'])    
    
    var =s.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))
    all_values.append(var)

    


ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values])

import pickle

with open('house_price_pred_ridge_model.pkl','rb') as f:
   chatgpt =  pickle.load(f)

price = chatgpt.predict(final_value)[0]


s.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))\

progress_bar = s.progress(0)
placeholder = s.empty()
placeholder.subheader('prediction Price')



if price > 0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    # st.subheader(body)
    placeholder.empty()

    s.success(body)
else:
    body = 'Invalid House features value'
    s.warning(body)
    
