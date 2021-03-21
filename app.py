from flask import Flask,render_template,url_for,redirect,request,session
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

with open("rdmodelserialized.pkl","rb") as f:
    model= pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods= ["GET","POST"])
def predict():
    if request.method == "POST":
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        pred_value = model.predict(final_features)
        
        if pred_value == 0:
            output= "Iris Setosa"
        elif pred_value == 1:
            output = "Iris Versicolar"
        else:
            output = "Iris Virginica"
        return render_template("index.html", prediction_text ='The Class of  Flower:  {}'.format(output))
    else:
        return redirect(url_for("index"))


    
if __name__ == "__main__":
    app.run(debug=True)