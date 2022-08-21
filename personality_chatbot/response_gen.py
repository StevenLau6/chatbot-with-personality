#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   text_summarizer.py
@Time    :   2022/02/23 16:48:55
@Author  :   Shuaiqi 
@Version :   1.0
@Contact :   shuaiqizju@gmail.com
@Desc    :   None
'''


import torch
import re
from transformers import Trainer, TrainingArguments

import pdb

MODEL_PATH_DICT={}
#MODEL_PATH_DICT['emo_response_gen']={}
MODEL_PATH_DICT['model_name']='t5_base_PELD'
MODEL_PATH_DICT['model_path']="/home/disk1/data/shuaiqi/emo_response_gen/model/t5_base/PELD/checkpoint-3906"


class emotional_response_gen(object):

    def __init__(
            self,
            generator_dict, 
            device = "cuda"   
    ):
        self.generator_dict = generator_dict
        #self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.torch_device = device if torch.cuda.is_available() else 'cpu'
        self.generator_name = generator_dict['model_name']
        self.generator_model_path = generator_dict['model_path']
        #self.input_text = input_text
        #self.section_name = section_name
        #self.summarizer_name = summarizer_name
        #self.output_len = output_len
        #self.input_truncate_len = input_len
        if "t5_base" in self.generator_name:
            #pdb.set_trace()

            from transformers import T5Tokenizer, T5ForConditionalGeneration#, Trainer, TrainingArguments

            self.tokenizer = T5Tokenizer.from_pretrained(self.generator_model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.generator_model_path).to(self.torch_device)

    
    def preprocess_input_text(self, input_str):
        
        input_str = str(input_str).strip().lower()
        input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
        input_str = input_str.encode('unicode_escape').decode('ascii')
        input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str)
        input_str = input_str.replace('\\ud835',' ').replace('    ',' ').replace('  ',' ').replace('  ',' ')
        return input_str
        
    def add_prefix(self, input_str,emotion,personality):
        prefix_str = "Dialog: " + "Emotion: " + emotion + ". Personality: " + personality + ". "
        prefix_added_str = prefix_str + input_str
        
        return prefix_added_str


    def response_generate(self, input_text, emotion = "joy", personality = "extraversion", output_max_len=70, output_min_len=1, input_max_length = 128):
        #output_text = ""
        cleaned_input_text = self.preprocess_input_text(input_text)
        prefix_added_text = self.add_prefix(cleaned_input_text,emotion,personality)

        test_batch = [prefix_added_text]

        #torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = self.tokenizer(test_batch,truncation=True, padding=True, max_length=input_max_length, return_tensors='pt').to(self.torch_device)

        predictions = self.model.generate(**inputs,max_length=output_max_len,min_length=output_min_len,num_beams=5,length_penalty=2.0,no_repeat_ngram_size=3)
        predictions = self.tokenizer.batch_decode(predictions)

        output_response_list = []
        for prediction in predictions:
            #cleaned_prediction = prediction.strip().replace('\n',' ').replace('\r',' ').replace('  ',' ').replace('  ',' ')
            cleaned_prediction = prediction.strip().replace('\n',' ').replace('\r',' ').replace('<pad>','').replace('</s> ','').replace('</s>','').replace('<s>','').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
            output_response_list.append(cleaned_prediction)
        #output_text = predictions[0].strip()
        #output_sent_list = output_text.split('. ')
        #for output_sent_id in range(len(output_sent_list)):
        #    output_sent_list[output_sent_id]=output_sent_list[output_sent_id].capitalize()

        #if output_text[-1]!='.':
        #    output_sent_list = output_sent_list[:-1]
        #output_text='. '.join(output_sent_list)+'.'

        #print(predictions)
        #print(output_text)

        #pdb.set_trace()
        response_text = output_response_list[0]
        return response_text