from flask import Flask, request, redirect, url_for, flash, jsonify 
import numpy as np 
import pickle as p 
import pandas as pd 
import json 
app = Flask(__name__) 
@app.route('/api', methods=['POST']) 
def makecalc(): 
    d = request.get_json(force=True) 
     
	#form a dictinary object d from json 
    modelfile = 'models/prediction.pickle' 
    model = p.load(open(modelfile, 'rb')) 
    prediction=model.predict(pd.DataFrame(d.values()).T).to_string().split("    ")[1]
    return jsonify(prediction) 
 
if __name__ == '__main__': 
    app.run() 
