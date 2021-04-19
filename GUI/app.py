from flask import Flask, render_template, request, redirect
from author import MVNB

app = Flask(__name__)
mvnb = MVNB()
mvnb.trainModel()


@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/identify', methods=['GET', 'POST'])
def identify():
  if request.method == 'POST':
    text = str(request.form['text'])
    result = mvnb.predict(text)
    return render_template('identify.html', result=result)
  else:
    return render_template('identify.html')


if __name__ == '__main__':
    app.run(debug=False)