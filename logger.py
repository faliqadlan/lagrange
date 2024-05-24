import logging

class Logger:
    def __init__(self):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Set the formatter for the handler
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

    def log_debug(self, message):
        self.logger.debug(message)

# Create an instance of the Logger class
logger = Logger()

# Log some messages
logger.log_debug("Debug message")
