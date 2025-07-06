from flask import Flask, request, jsonify
from mlops import train_model, predict_spam, get_best_params

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    acc = train_model()
    return jsonify({'message': 'Model trained successfully.', 'accuracy': acc}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Text field is required.'}), 400
    prediction = predict_spam(data['text'])
    return jsonify({'prediction': prediction}), 200

@app.route('/best_model_parameter', methods=['GET'])
def best_params():
    params = get_best_params()
    return jsonify(params), 200

if __name__ == '__main__':
    app.run(debug=True)
