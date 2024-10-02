# app/__init__.py

from flask import Flask, g
import mysql.connector
from config import config

def create_app():
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config)

    # Register Blueprints
    from app.views.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Register auth blueprint
    from app.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

     # Set up database connection
    @app.before_request
    def before_request():
        g.db = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )

    @app.teardown_request
    def teardown_request(exception):
        db = getattr(g, 'db', None)
        if db is not None:
            db.close()

    return app
