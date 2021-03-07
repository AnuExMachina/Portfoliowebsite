from django.shortcuts import render
from django.http import JsonResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response

import pandas as pd
import joblib
import os
import torch 
from .models.titanic import TitanicNN

# Create your views here.
@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List': '/model-list/',
        'Titanic': '/titanic/',
    }
    return Response(api_urls)

@api_view(['POST'])
def apiTitanic(request):
    print(request.data)
    print(os.getcwd())
    data = request.data
    data['Pclass'] = int(data['Pclass'])
    data['Age'] = int(data['Age']) 
    data['Siblings/Spouses Aboard'] = int(data['Siblings/Spouses Aboard']) 
    data['Parents/Children Aboard'] = int(data['Parents/Children Aboard']) 
    data['Fare'] = float(data['Fare'])  
    label_encoder = joblib.load('api/artifacts/titanic/label_encoder.pkl')
    one_hot_encoder = joblib.load('api/artifacts/titanic/one_hot_encoder.pkl')
    scaler = joblib.load('api/artifacts/titanic/scaler.pkl')
    X = pd.DataFrame.from_dict([data])
    print(X)
    transformed = pd.DataFrame(one_hot_encoder.transform(X['Pclass'].to_numpy().reshape(-1, 1)), columns=one_hot_encoder.get_feature_names(['Pclass']))
    X = pd.concat([X.drop('Pclass', axis=1), transformed], axis=1)
    X['Sex'] = label_encoder.transform(X['Sex'])
    X = scaler.transform(X)
    X = torch.Tensor(X)
    model = TitanicNN.load_from_checkpoint('api/artifacts/titanic/model.ckpt')
    model.eval()
    X = model(X)
    print(X)


    return Response(X)




# @api_view(['GET'])
# def taskList(request):
#     tasks = Task.objects.all()
#     serializer = TaskSerializer(tasks, many=True)
#     return Response()

# {
# "a":5,
# "b":3,
# "c":4
# }



#  {
#  "Pclass": 1,
#  "Sex": "male",
#  "Age": 30,
#  "Siblings/Spouses Aboard": 1,
#  "Parents/Children Aboard": 1,
#  "Fare": 7.2500
#  }