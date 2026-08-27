from datetime import datetime


def convert_date_to_timestamp(date_str: str) -> int:
    dt = None

    if not dt:  # LoCoMo
        try:
            dt = datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
        except:
            dt = None

    if not dt:  # LongMemEval
        try:
            clean_str = date_str.split(" (")[0] + " " + date_str.split(") ")[1]
            dt = datetime.strptime(clean_str, "%Y/%m/%d %H:%M")
        except:
            dt = None

    if not dt:  # Mem Summarize By Date
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            dt = None

    if not dt:  # Shopping Companion
        try:
            dt = datetime.strptime(date_str, "%Y/%m/%d (%a)")
        except:
            dt = None

    if not dt:  # Shopping Companion
        try:
            dt = datetime.strptime(date_str, "%Y/%m/%d")
        except:
            dt = None

    if not dt:
        return 0

    return int(dt.timestamp())


def is_rubbish_kv(key: str, value: str) -> bool:
    rubbish_words = {
        "units_hb",
        "multi-pack",
        "multi pack",
        "multipack",
        "pack_type",
        "nobrand",
        "no brand",
        "notbrand",
        "not brand",
        "notbranded",
        "not branded",
        "na",
        "n/a",
        "null",
        "no",
        "nospecified",
        "no specified",
        "notspecified",
        "not specified",
        "other",
        "others",
        "unknown",
        "miscellaneous",
        "misc",
        "new",
    }
    if key.lower() in rubbish_words or value.lower() in rubbish_words:
        return True

    return False
