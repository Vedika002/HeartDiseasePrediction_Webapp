import numpy as np
from flask import Flask, request, render_template
import pickle

app= Flask(__name__)

model= pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('ind.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    features = np.array(int_features)
    reshaped=features.reshape(1,-1)
    prediction = model.predict(reshaped)
    if prediction == 0:
        predicts=0
    else:
        predicts=1    
    return render_template('ind.html', prediction=predicts)

if __name__ == "__main__":
    app.run()