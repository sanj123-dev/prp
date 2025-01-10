from flask import Flask, request, jsonify
from BertMini import SMSCategorization


app = Flask(__name__)

categorizer = SMSCategorization()



@app.route("/ping")
def ping():
    return jsonify({"status": True, "message": "it work fine....."})
    


@app.route('/categorize_sms', methods=['POST'])
def categorize_sms():
    try:
        data = request.get_json()
        
        sms_messages = data.get('sms_messages', [])
        
        if not isinstance(sms_messages, list) or not sms_messages:
            return jsonify({"error": "Invalid input, provide a list of SMS messages"}), 400
        
        results = categorizer.categorize_sms(sms_messages)
        
        response = [{"sms": result["sms"], "category": result["category"]} for result in results]
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()

