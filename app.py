from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time

app = Flask(__name__)
CORS(app)

@app.route('/price', methods=['POST'])
def predict_price():
    data = request.json
    
    random_price = round(random.uniform(100, 500), 2)

    time.sleep(3)

    response_data = {
        "received_params": data,
        "price": random_price
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)