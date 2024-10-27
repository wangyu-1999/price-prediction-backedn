from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
from get_coordinates import get_coordinates

app = Flask(__name__)
CORS(app)

@app.route('/price', methods=['POST'])
def predict_price():
    data = request.json
    
    random_price = round(random.uniform(100, 500), 2)

    neighbourhood = data.get("Host_neighbourhood", "Downtown")
    
    response_data = {
        "received_params": data,
        "price": random_price,
        "neighbours": get_coordinates(neighbourhood)
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
