from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Call your model function here using the input_text variable
        # Display the result in the template
        return render_template('result.html', input_text=input_text)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
