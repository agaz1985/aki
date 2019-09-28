import logging

logger = logging.getLogger('aki_logger')
logger.setLevel(level=logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)
