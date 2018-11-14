from datetime import datetime, timedelta, date
from dateutil import parser
import pytz

settings = {}
settings['TIMEZONE'] = pytz.timezone('America/Chicago')


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
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"]).\
            strftime('%Y-%m-%d %H:%M:%S%z')
    elif len(str(abs(unixtime))) == 10:
        return datetime.utcfromtimestamp(unixtime).\
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"]).\
            strftime('%Y-%m-%d %H:%M:%S%z')
    
    