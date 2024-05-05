"""
This script runs the FlaskWebProject1 application using a development server.
"""

from os import environ
from FlaskWebProject1 import app

from gevent import pywsgi
if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    #server = pywsgi.WSGIServer(('0.0.0.0',5000),app)
    #server.serve_forever()
    app.run(HOST, PORT)
