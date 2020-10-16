from flask import Flask,render_template,request
import pickle
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner

###Loading model and cv
cv = pickle.load(open('cv.pkl','rb'))
model = pickle.load(open('spam.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        mail = request.form['email']
        data = [mail]
        pred = model.predict_batch(data)
        return render_template('result.html',prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)    
