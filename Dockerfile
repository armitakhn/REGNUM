FROM python:3.8-slim

WORKDIR /numamie
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev

RUN apt-get install -y openjdk-11-jdk

RUN pip install --no-cache-dir --upgrade pip

COPY ./requirements.txt /numamie

RUN pip install --no-cache-dir -r requirements.txt


COPY ./src /numamie/src
COPY data/rule_miners/amie_jar /numamie/../amie_jar


EXPOSE 1030

#CMD ["gunicorn"  , "-b", "0.0.0.0:1030", "src.wsgi:application"]
CMD python src/web_service.py
