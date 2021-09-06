#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle
from joblib import load


# In[ ]:


app = Flask(__name__)
model = pickle.load(open('irish_model.pkl', 'rb'))


# In[ ]:


@app.route('/')
def home():
    return render_template('index_new.html')


# In[ ]:


@app.route('/predict', methods=['POST'])
def predict():
    int_features = np.array([float(x) for x in request.form.values()])
    #print(int_features)
    final_features = int_features.reshape(1,-1)
    print(type(final_features))
    scalerX = load(open('scaler_X.pkl', 'rb'))
    final_features_scaled = scalerX.transform(final_features)
    prediction = model.predict(final_features_scaled)
    #print(type(prediction))
    #print(prediction.shape)
    if prediction.item(0) == 0:
        pred_final = 'Irish-setosa'
    elif prediction.item(0) == 1:
        pred_final = 'Irish-versicolor'
    else:
        pred_final = 'Irish-virginica'
    return render_template('index.html', prediction_text=pred_final)


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:
