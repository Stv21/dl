from flask import Flask, send_file

app = Flask(__name__)

@app.route('/and')
def and_page():
    return send_file('and.html')

if __name__ == '__main__':
    app.run(debug=True)