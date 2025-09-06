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

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple",)
text = "We are looking for someone with Python, SQL, and React skills."
ner_results = ner(text)
#Each keyword in the listing above (python, sql and react) are all tagged as miscellaneous since they dont fit into the pre trained labels of person, location, or organization
print(ner_results)


#Need to prepare a custom dataset of job listings that the model can train on
