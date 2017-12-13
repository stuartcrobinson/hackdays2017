
from flask import Flask
from flask import request



app = Flask(__name__)

@app.route('/test')
def index6():
    return "hi"


if __name__ == '__main__':
    app.run(debug=False)    
