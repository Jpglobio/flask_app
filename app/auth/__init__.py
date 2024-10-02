from flask import Blueprint

auth = Blueprint('auth', __name__)

from . import routes  # Import routes to register the routes with the blueprint
