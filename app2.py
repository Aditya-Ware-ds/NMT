from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, MarianMTModel

app = Flask(__name__)

# Load the translation models
def load_translation_model(src_lang, trg_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@app.route('/')
def index():
    return render_template('alpha3.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    input_text = data.get('text')
    target_lang = data.get('lang')  

    if not input_text or not target_lang:
        return jsonify({'error': 'Missing text or target language'}), 400

    
    src_lang = "en"

    try:
        
        model, tokenizer = load_translation_model(src_lang, target_lang)

        
        batch = tokenizer([input_text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
