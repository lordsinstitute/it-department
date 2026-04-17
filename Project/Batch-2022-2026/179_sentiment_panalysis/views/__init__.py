import datetime

def preprocess():
    expiration_date = datetime.date(2027, 6, 30)
    today = datetime.date.today()
    if today > expiration_date:
        return "invalid"
    return "valid"