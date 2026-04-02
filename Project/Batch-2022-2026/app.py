from flask import *
import views.adminbp, views.userbp
app = Flask(__name__)
app.secret_key = "abc"
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)

@app.route('/')
def home():
    return render_template("home.html")


if __name__ == '__main__':
    app.run()


"""
import pickle
from flask import *
import views.adminbp, views.userbp
app = Flask(__name__)
app.secret_key = "abc"
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)

import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

Pkl_Filename = "rf_tuned.pkl"
with open(Pkl_Filename, 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template("home.html")





if __name__ == '__main__':
    app.run(debug=True)
"""