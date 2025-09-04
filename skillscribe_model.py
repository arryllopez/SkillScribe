import torch 
import bitsandbytes as bnb
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
from transformers import BitsAndBytesConfig
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER",) 

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple",)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)
