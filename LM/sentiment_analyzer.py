from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Thaweewat/wangchanberta-hyperopt-sentiment-01")
model = AutoModelForSequenceClassification.from_pretrained("Thaweewat/wangchanberta-hyperopt-sentiment-01")

