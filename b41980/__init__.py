# Importing modules
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


# Initializing our flask app and configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'c211d64055079c4e'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///b41980rf.db'
db = SQLAlchemy(app)

from b41980 import routes
