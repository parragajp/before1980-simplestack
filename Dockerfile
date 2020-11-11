FROM tiangolo/uwsgi-nginx-flask:python3.8

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY main.py __init__.py rf_before1980.joblib home.md /app/

CMD [ "python3.8", "main.py" ]