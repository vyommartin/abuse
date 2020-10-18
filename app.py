from flask import Flask,render_template,request
import pickle
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import aiohttp
import asyncio
import uvicorn
from pathlib import Path
import os
from os import *
import numpy
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import torch

path = Path(__file__).parent

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)
                
pretrained_link = "https://www.googleapis.com/drive/v3/files/1-00f28mlffM2uPJVJDY94K1aOy9LfJw1?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA"
modelname = 'pytorch_model.bin'
                
logger = logging.getLogger()

async def setup_learner():
    await download_file(pretrained_link, path / modelname)
    try:
        data_bunch = BertDataBunch(path, path,
                           tokenizer = path,
                           train_file = None,
                           val_file = None,
                           label_file = 'l2.csv',
                           batch_size_per_gpu = 120,
                           max_seq_length = 40,
                           multi_gpu = False,
                           multi_label = False,
                           model_type = 'bert') 
        
        learner = BertLearner.from_pretrained_model(data_bunch, 
                                            pretrained_path = path,
                                            metrics = [],
                                            device = 'cpu',
                                            logger = None,
                                            output_dir = None,
                                            is_fp16 = False)
        return learner 
                

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        mail = request.form['email']
        data = [mail]
        pred = learner.predict_batch(data)
        return render_template('result.html',prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)    
