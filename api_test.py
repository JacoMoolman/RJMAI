from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test_endpoint():
    # Handle both JSON and form data
    if request.is_json:
        data = request.json
        text = data.get('text', '')
    else:
        # Try to get form data
        text = request.form.get('text', '')
        
        # If no form data, try to parse raw data
        if not text and request.data:
            try:
                # Try to decode raw data
                raw_data = request.data.decode('utf-8')
                print(f"Raw data received: {raw_data}")
                text = raw_data
            except:
                text = "Unable to parse request data"
    
    print(f"Received text: {text}")
    return jsonify({"response": f"Got it: {text}"})

if __name__ == '__main__':
    print("API test server running on http://localhost:5000")
    print("Send a POST request to http://localhost:5000/test with data")
    app.run(debug=True)