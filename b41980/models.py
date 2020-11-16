from b41980 import db


class House(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    livearea = db.Column(db.Integer, nullable=False)
    bdrms = db.Column(db.Integer, nullable=False)
    baths = db.Column(db.Integer, nullable=False)
    one_story = db.Column(db.Integer, nullable=False)
    att_garage = db.Column(db.Integer, nullable=False)
    basement = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"House {self.livearea} {self.bdrms} {self.baths} {self.one_story} {self.att_garage} {self.basement}"
