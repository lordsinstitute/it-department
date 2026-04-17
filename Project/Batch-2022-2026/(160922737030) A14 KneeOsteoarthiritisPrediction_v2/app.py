from flask import *
import views.adminbp, views.userbp
app = Flask(__name__)
app.secret_key = "abc"
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run()
