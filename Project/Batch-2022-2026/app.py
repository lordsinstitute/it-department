from flask import *
import views.adminbp, views.userbp



app = Flask(__name__)
app.register_blueprint(views.adminbp.admin_bp)
app.register_blueprint(views.userbp.user_bp)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')




if __name__ == '__main__':
    app.run()
