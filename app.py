#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle


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
    int_features = [float(x) for x in request.form.values()]
    #print(int_features)
    final_features = [np.array(int_features)]
    #print(type(final_features))
    sc = StandardScaler()
    final_features = sc.fit_transform(final_features)
    prediction = model.predict(final_features)
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




