import os
from datetime import datetime, timedelta, date
from dateutil import parser
import pytz

settings = {}
settings['TIMEZONE'] = pytz.timezone('America/Chicago')



def create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def lprint(logfile, *argv): # for python version 3

    """ 
    Function description: 
    ----------
        Save output to log files and print on the screen.
    Function description: 
    ----------
        var = 1
        lprint('log.txt', var)
        lprint('log.txt','Python',' code')
    Parameters
    ----------
        logfile:                 the log file path and file name.
        argv:                    what should 
        
    Return
    ------
        none
    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
    """

    # argument check
    if len(argv) == 0:
        print('Err: wrong usage of func lprint().')
        sys.exit()

    argAll = argv[0] if isinstance(argv[0], str) else str(argv[0])
    for arg in argv[1:]:
        argAll = argAll + (arg if isinstance(arg, str) else str(arg))
    
    print(argAll)

    with open(logfile, 'a') as out:
        out.write(argAll + '\n')


def unixtime_to_datetime(unixtime):
    if len(str(abs(unixtime))) == 13:
        return datetime.utcfromtimestamp(unixtime/1000).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])
    elif len(str(abs(unixtime))) == 10:
        return datetime.utcfromtimestamp(unixtime).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])


def datetime_to_unixtime(dt):
    '''
    Convert Python datetime object (timezone aware)
    to epoch unix time in millisecond
    '''
    return int(1000 * dt.timestamp())


def parse_timestamp_tz_naive(string):
    STARTTIME_FORMAT_WO_CENTURY = '%m/%d/%y %H:%M:%S'
    STARTTIME_FORMAT_W_CENTURY = '%m/%d/%Y %H:%M:%S'
    try:
        dt = datetime.strptime(string, STARTTIME_FORMAT_WO_CENTURY)
    except:
        dt = datetime.strptime(string, STARTTIME_FORMAT_W_CENTURY)

    return dt


def datetime_str_to_unixtime(string):
    return datetime_to_unixtime(parse_timestamp_tz_aware(string))


def parse_timestamp_tz_aware(string):
    return parser.parse(string)


def unixtime_to_datetime(unixtime):
    if len(str(abs(unixtime))) == 13:
        return datetime.utcfromtimestamp(unixtime/1000).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])
            # strftime('%Y-%m-%d %H:%M:%S%z')
    elif len(str(abs(unixtime))) == 10:
        return datetime.utcfromtimestamp(unixtime).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])
            # strftime('%Y-%m-%d %H:%M:%S%z')
    

def string_to_datetime(startDatetime, endDatetime):
    # startDatetime = '06-27-17_11'
    # endDatetime = '06-27-17_15'
    start = datetime.strptime(startDatetime, '%m-%d-%y_%H_%M_%S')
    end = datetime.strptime(endDatetime, '%m-%d-%y_%H_%M_%S')
    return start, end


def datetime_to_foldername(dt):
    return dt.strftime('%m-%d-%y')

def datetime_to_filename(dt):
    return dt.strftime('%m-%d-%y_%H.csv')

    