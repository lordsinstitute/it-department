from flask import Flask, render_template, request
import views.adminbp
import views.userbp
import logging
import re
from classifier import ClassiferSingleton

app = Flask(__name__)
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)


def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_classifier():
    cs = ClassiferSingleton()
    cs.set_paths(model_path='finalized_model.sav', vectorizer_path='vectorizer.sav')


def extract_video_id(url):
    if not url:
        return None
    match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    match = re.search(r'/shorts/([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def sentiment():
    link = False
    error = None
    if request.method == "POST":
        raw_url = request.form.get('videoQuery', '').strip()
        video_id = extract_video_id(raw_url)
        if video_id:
            try:
                full_url = f"https://www.youtube.com/watch?v={video_id}"
                ClassiferSingleton().make_analysis(full_url)
                link = video_id
            except Exception as e:
                error = f"Could not fetch comments: {str(e)}"
                link = video_id
        else:
            error = "Invalid YouTube URL. Please paste a valid link."
    return render_template('index.html', link=link, error=error)


if __name__ == '__main__':
    configure_logging()
    load_classifier()
    app.run(debug=True)