from flask import Flask, render_template, request
import views
from views import adminbp, userbp


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "abc"
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
