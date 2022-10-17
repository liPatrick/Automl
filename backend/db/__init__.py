from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from flask_sqlalchemy import SQLAlchemy

def create_mock_db():
    engine = create_engine("postgresql://localhost/yugo")
    if not database_exists(engine.url):
        create_database(engine.url)

    return database_exists(engine.url)

class MLDB:
    def __init__(self, create_engine=False):
        # This DB is the global db object used in Flask applications.
        # It can be imported from here to get access to a request-scoped
        # session from anywhere in the codebase. Flask-SQLAlchemy takes care
        # that each session is unique to a thread, but the same session
        # should not be shared by two threads.
        if create_engine:
            create_mock_db()

        self.flask_sqlalchemy_db = SQLAlchemy()
        self.declarative_base = self.flask_sqlalchemy_db.Model
        self.app_session = self.flask_sqlalchemy_db.session

ml_db = MLDB()