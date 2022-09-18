from flask import Flask


app = Flask(__name__)
app.config['SECRET_KEY'] = '75f99765ab52e13194e112135e67e46cbac2d56f'


from application import routes