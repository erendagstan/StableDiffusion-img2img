from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/task1', methods=['POST'])
def task1_api():
    try:
        user_image = request.files['user_image']
        user_prompt = request.form['user_prompt']
        user_color = request.form['user_color']
        # Perform Task 1
        from img2img import task1_imp
        images = task1_imp(user_image, user_prompt, user_color)
        return jsonify(status='success', message='Task 1 completed successfully', result=str(images))
    except Exception as e:
        return jsonify(status='error', message=str(e))


@app.route('/task2', methods=['POST'])
def task2_api():
    try:
        user_image = request.files['user_image']
        logo_path = request.form['logo_path']
        button_color = request.form['button_color']
        punchline_text = request.form['punchline_text']
        button_text = request.form['button_text']
        # Perform Task 2
        from img2img import task2_imp
        new_image = task2_imp(user_image, logo_path, button_color, punchline_text, button_text)
        return jsonify(status='success', message='Task 2 completed successfully', result=str(new_image))

    except Exception as e:
        return jsonify(status='error', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)