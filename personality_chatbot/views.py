#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   views.py
@Time    :   2022/08/21 11:13:20
@Author  :   sq 
@Version :   1.0
@Desc    :   None
'''

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import time
import pandas as pd
import numpy as np

import traceback

#from personality_chatbot.response_gen import emotional_response_gen
from .response_gen import emotional_response_gen

#from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pdb

OUTPUT_MAX_LEN=70
OUTPUT_MIN_LEN=1
INPUT_MAX_LENGTH = 128

bert_emotion_tokenizer = AutoTokenizer.from_pretrained("/home/zhiyuan/aaai'23/src/bert_emotion_classification")  #bert_emotion_classification
bert_emotion_model = AutoModelForSequenceClassification.from_pretrained("/home/zhiyuan/aaai'23/src/bert_emotion_classification", num_labels=6)   #"bert_emotion_classification"


def emotionInferenceBERT(input_sentence,bert_tokenizer,bert_model):
    inputs = bert_tokenizer(input_sentence, return_tensors = "pt")
    outputs = bert_model(**inputs)
    label_idx = torch.argmax(outputs[0]).item()
    emotion_mapping = {4: "sadness", 2: "joy", 1: "fear", 0: "anger", 5: "surprise", 3: "love"}
    #print(label_idx)
    #pdb.set_trace()
    return emotion_mapping[label_idx]
    

def chatbot_emotion_gen(input_str):
    return "joy"

@csrf_exempt
def get_response(request):

    input_str = request.POST.get('input_str')
    input_personality = request.POST.get('personality')
    #nput_emotion = request.POST.get('emotion')

    # user emotion classification
    #input_emotion = emotionInferenceBERT(input_str)
    input_emotion = emotionInferenceBERT(input_str,bert_emotion_tokenizer,bert_emotion_model)

    # chatbot emotion predict
    response_emotion = chatbot_emotion_gen(input_str)

    # response generation
    generator_dict={}
    generator_dict['model_name']='t5_base_PELD'
    generator_dict['model_path']="/home/disk1/data/shuaiqi/emo_response_gen/model/t5_base/PELD/checkpoint-3906"

    emo_generator = emotional_response_gen(generator_dict,device = "cuda:1" )

    response_text = emo_generator.response_generate(input_str, emotion = response_emotion, personality = input_personality, output_max_len=OUTPUT_MAX_LEN, output_min_len=OUTPUT_MIN_LEN, input_max_length = INPUT_MAX_LENGTH)

    response_dict = {}
    response_dict["input_str"] = input_str
    response_dict["response_text"] = response_text
    response_dict["response_emotion"] = response_emotion
    response_dict["input_emotion"] = input_emotion
    print(response_dict)
    response_json = json.dumps(response_dict)

    return HttpResponse(response_json)

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
