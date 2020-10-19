from flask import Flask,render_template,request
import pickle
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import aiohttp
import asyncio
from pathlib import Path
import os
from os import *
import numpy
from io import BytesIO
import torch

path = Path('/opt/render/project/src/')

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)
                
#l2link = 'https://www.googleapis.com/drive/v3/files/1YY-85GBK_TRK50B5YqXogn6d69qfmCdc?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA'
pretrained_link = "https://www.googleapis.com/drive/v3/files/1-00f28mlffM2uPJVJDY94K1aOy9LfJw1?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA"
#vocablink = "https://www.googleapis.com/drive/v3/files/1-BYV3NlKGhD32Srbb0fe15WAlkLTMCVh?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA"
#sptokenlink = "https://www.googleapis.com/drive/v3/files/1-2Zf_PjqNLeo0QMLmAGLmhMJ5GTxJux6?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA"
#tokenlink = "https://www.googleapis.com/drive/v3/files/1-5Bx9rIaq24_3niulnuvABK0wDbl4Bzu?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA"
#configlink = "https://www.googleapis.com/drive/v3/files/1-1XqYH-5DYKKNiTM8h5WTPQFRNAZtVAy?alt=media&key=AIzaSyArebv-g7_CgQUjKftzGkgeHhtHivaR4TA"

modelname = 'pytorch_model.bin'
#vocab = 'vocab.txt'
#sptoken = 'special_tokens_map.json'
#token = 'tokenizer_config.json'
#config = 'config.json'
#l2 = 'l2.csv'


    
async def setup_learner():
    await download_file(pretrained_link, path / modelname),
    #await download_file(vocablink, path / vocab),
    #await download_file(sptokenlink, path / sptoken),
    #await download_file(tokenlink, path / token),
    #await download_file(configlink, path / config),
    #await download_file(l2link, path / l2)
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
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise
                
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()
                
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
