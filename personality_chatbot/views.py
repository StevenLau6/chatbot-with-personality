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

OUTPUT_MAX_LEN=70
OUTPUT_MIN_LEN=1
INPUT_MAX_LENGTH = 128


def emotionInferenceBERT(input_str):
    return "joy"

def chatbot_emotion_gen(input_str):
    return "joy"

@csrf_exempt
def get_response(request):

    input_str = request.POST.get('input_str')
    input_personality = request.POST.get('personality')
    #nput_emotion = request.POST.get('emotion')

    # user emotion classification
    input_emotion = emotionInferenceBERT(input_str)

    # chatbot emotion predict
    response_emotion = chatbot_emotion_gen(input_str)

    # response generation
    generator_dict={}
    generator_dict['model_name']='t5_base_PELD'
    generator_dict['model_path']="/home/disk1/data/shuaiqi/emo_response_gen/model/t5_base/PELD/checkpoint-3906"

    emo_generator = emotional_response_gen(generator_dict,device = "cuda:1" )

    response_text = emo_generator.response_generate(input_str, emotion = response_emotion, personality = input_personality, output_max_len=OUTPUT_MAX_LEN, output_min_len=OUTPUT_MIN_LEN, input_max_length = INPUT_MAX_LENGTH)

    response_dict = {}
    response_dict["response_text"] = response_text
    response_dict["response_emotion"] = response_emotion
    response_dict["input_emotion"] = input_emotion
    #print(response_dict)
    response_json = json.dumps(response_dict)

    return HttpResponse(response_json)

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
