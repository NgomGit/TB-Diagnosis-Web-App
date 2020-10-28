from django.shortcuts import render,redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import login
from django.urls import reverse

from .forms import UploadFileForm
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from glob import glob
import os


from tb_app.forms import UserForm

models = []


def index(request):

    filename = "placeholder.png"

    if request.method == 'POST':
        if request.FILES != {}:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            result, tb_score, ntb_score, threshold = DSGU(filename)
            #load_models()
            return render(request, 'tb_app/index.html', {
                'result': result,
                'tb_score':tb_score,
                'ntb_score':ntb_score,
                'threshold':0.3,
                'filename':filename,
                'progress_value_tb':int(np.round(tb_score*100)),
                'progress_value_ntb':int(np.round(ntb_score*100)),
            })
        else :
            return render(request, 'tb_app/index.html', {
                'error': "Please choose an image",
            })
    else :
        if(models == []):
            load_models()
    
        
       
    return render(request,'tb_app/index.html',{'filename':filename})

     #else :
          #  return render(request, 'tb_app/index.html', {
             #   'error': "Error while uploading the file "
           # })

def load_models():
    global models

    vgg_model = load_model('vgg_train_on_old_data')
    cnn_gen_model = load_model('tb_app/models/cnn_gen_model')
    attention_cnn_gen = load_model('tb_app/models/cnn_gen_attention')
    cnn_train_old_data = load_model('tb_app/models/cnn_train_old_data')
    
    models = [vgg_model,cnn_gen_model,attention_cnn_gen,cnn_train_old_data]

def get_models_scores():
    '''
    this function return basically two list the models
     and theire scores
    respectively
    '''
    
    vgg_scores = np.array ([0.95535195,0.7654883])

    cnn_gen_scores = np.array ([0.8863358,0.9293535])

    attention_cnn_gen_scores = np.array ([0.8282092751526251,0.8510600569881971])

    cnn_train_old_data_scores = np.array ([0.9308654,0.86002976])

    #models = load_models()

    return models ,[vgg_scores, cnn_gen_scores,attention_cnn_gen_scores,cnn_train_old_data_scores]

def predict_classes(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities 
    """
    x = image.img_to_array(img)
    x = x/255 #rescaling the image
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds[0]
  
def class_decision_2(model_preds,scores):
    '''Args : 
    Take the predictions of the models and its scores,
    return :  '''
    a  = 0
    i = 0
    tb_prob = 0 
    ntb_prob = 0
    for pred in model_preds :
        ntb_score,tb_score = get_labels_probs(scores[i],pred)
        ntb_prob += ntb_score
        tb_prob += tb_score
        i +=1
    
    ntb_score , tb_score = normalized_result(ntb_prob/4,tb_prob/4 )
    return ntb_score , tb_score,(ntb_prob/4-tb_prob/4)

def normalized_result(score_a , score_b):
    result_a = np.exp(score_a)/(np.exp(score_a) + np.exp(score_b))
    result_b = np.exp(score_b)/(np.exp(score_a) + np.exp(score_b))

    return np.round(result_a,3), np.round(result_b,3)

def get_labels_probs(scores,predic):
    if len (predic) > 1:
        a =  predic[0]
        b = predic[1]
    else :
        b = predic[0]
        a = 1 - predic[0]

    ntb_score = scores[0] * a   - ([1,1]- scores) [1]
    tb_score = scores[1] * b - ([1,1]- scores) [0] 

    return ntb_score ,tb_score

def DSGU(img_name):
    result = ""
    img = load_image(img_name)
    print('Image Type', img)
    models, scores = get_models_scores()
    predictions = []
    for model in models:
        model_preds = predict_classes(model,img)
        predictions.append(model_preds)
  
    ntb_score,tb_score,threshold = class_decision_2(predictions,scores)
  
   
    
    if tb_score > ntb_score:
        result = " TB Positive"
    else:
        if  threshold < 0.3: # the models is not that sure of the person has or not tb
            result = " TB Positive"
        else:
            result = " TB Negative"

    

    return result, tb_score, ntb_score , threshold
    
def load_image(img_name):
   img = image.load_img('tb_app/static/media/'+img_name, target_size=(299, 299))
   return img


def dashboard(request):
    return render(request, "tb_app/dashboard.html")

def register(request):
    if request.method == "GET":
        return render(
            request, "tb_app/register.html",
            {"form": UserForm}
        )
    elif request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect(reverse("dashboard"))