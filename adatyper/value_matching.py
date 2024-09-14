import json



def check_latitude(value: str):
    try:
        lat = abs(int(value))
        if lat <= 90:
            return True
        return False
    except:
        return False


def check_longitude(value: str):
    try:
        lon = abs(int(value))
        if lon <= 180:
            return True
        return False
    except:
        return False


def check_json(value: str):
    try:
        decoded = json.JSONDecoder().decode(value)
        return True
    except:
        return False


def check_us_postal_code(value: str):
    try:
        if len(str(value)) == 5 and (
            isinstance(value, int) or isinstance(int(value), int)
        ):
            return True
        return False
    except:
        return False


def get_type_function_dict():
    type_functions_dict = {
        "json": [check_json],
        "postal code": [check_us_postal_code],
        "latitude": [check_latitude],
        "longitude": [check_longitude],
    }

    return type_functions_dict


def get_regular_expression_dict():

    # TODO: fetch WikiData classes and properties, that have the property: https://www.wikidata.org/wiki/Property:P1793 (format as a regular expression)
    with open("adatyper/type_regex.json") as f:
        type_regex_dict = json.load(f)

    return type_regex_dict
