"""Sample file with a few intentional issues for local analysis."""

import math
import os


def add_item(item, items=[]):
    items.append(item)
    return items


def safe_divide(left, right):
    try:
        return left / right
    except:
        return None


def almost_pi():
    value = math.pi
    if value == None:
        return 0
    unused = 42
    return value


def run_user_code(text):
    return eval(text)
