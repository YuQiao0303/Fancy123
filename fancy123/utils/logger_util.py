import time
import datetime
import logging
import os

def get_logger(logfile = None):
    def beijing(sec):
        if time.strftime('%z') == "+0800":
            return datetime.datetime.now().timetuple()
        return (datetime.datetime.now() + datetime.timedelta(hours=8)).timetuple()


    # https://blog.csdn.net/dcrmg/article/details/88800685?spm=1001.2101.3001.6661.1&
    # 1. create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)  # Log level

    # 2. create handler
    if logfile is None:
        # log_path = os.path.dirname(os.getcwd()) + '/Logs/'
        log_path = os.path.join(os.getcwd(), 'Logs')
        logfile = os.path.join(log_path,'log.log')
    else:
        log_path, _ = os.path.split(logfile)
    os.makedirs(log_path, exist_ok=True)
    # print('os.getcwd()',os.getcwd())
    # print('log_path',log_path)
    print('logfile',logfile)

    file_handler = logging.FileHandler(logfile, mode='a+')
    file_handler.setLevel(logging.INFO)  

    # 3. define output fommer for handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    formatter.converter = beijing # beijing time # if you use other time, modify this

    # 4. add handler into logger
    logger.addHandler(file_handler)

    # 5. if need to output to terminal , define a streamhandler 
    # terminal_formatter = logging.Formatter("%(asctime)s -  %(message)s")
    # terminal_formatter.converter = beijing
    # print_handler = logging.StreamHandler()  
    # print_handler.setFormatter(terminal_formatter)  
    # print_handler.setLevel(logging.WARNING)  
    # logger.addHandler(print_handler)
    return logger