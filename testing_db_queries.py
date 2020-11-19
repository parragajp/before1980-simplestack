from b41980 import db
from b41980.models import House

db.drop_all()
db.create_all()

db.Query.first()
House.query.all()
query3 = House.query.order_by(House.id.desc()).first()
query3.livearea
