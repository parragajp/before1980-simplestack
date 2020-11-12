from b41980 import db


class House(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    livearea = db.Column(db.Integer, nullable=False)
    stories = db.Column(db.Integer, nullable=False)
    bdrms = db.Column(db.Integer, nullable=False)
    baths = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"House {self.livearea} {self.stories} {self.bdrms} {self.baths}"
