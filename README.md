# before1980-simplestack
End to end machine learning project that predicts whether a house was built before 1980. Used flask to create api services and website. The initial app was put on docker hub. The most recent version of the app, which is far improved, has not been put in a docker image or docker hub. You can run it locally though. Instructions are below.

## Helpful commands

### How to spin up container, if you have docker already installed

Note: It will download the image if you don't already have it. ~1 Gb.
_This is the bare bones outdated version of the app._

```
docker run -p 1001:80 brandonjenkins/simple_house_mod:first
```

### How to work with database after creating tables with classes
```python
# Creating sqllite table
from b41980 import db
db.create_all()

# Writing new data to table
from b41980.models import House
house1 = House(livearea=2200, stories=2, bdrms=3, baths=1)  # Automatically created id
db.session.add(house1)
db.session.commit()

# Querying the data
House.query.all()  # Only works if __repr__ is set
House.query.first()
House.query.get(1)  # Grabs row by id

# Drop the table
db.drop_all()
```

### How to run most updated version of app locally

_You will need to have all the necessary packages installed_

_Make sure your current directory is at the project level_

```
python3 run.py
```
