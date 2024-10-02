from flask import g

class Student:
    def __init__(self, id=None, fname=None, major=None, scholarship=None, amount=None, details=None):
        self.id = id
        self.fname = fname
        self.major = major
        self.scholarship = scholarship
        self.amount = amount
        self.details = details
