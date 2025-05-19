
from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# Define the path to the local model directory
model_dir = os.path.join(os.path.dirname(__file__), "fake_news_bert_model")

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Prediction function
def predict_news(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
    return "Real" if prediction == 0 else "Fake"

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_text = ""
    if request.method == 'POST':
        input_text = request.form['headline']
        if input_text.strip():
            prediction = predict_news(input_text)
    return render_template('index.html', prediction=prediction, headline=input_text)

if __name__ == '__main__':
    app.run(debug=True)
