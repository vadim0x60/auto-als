FROM python:3.9.16-bullseye

ADD . /app
WORKDIR /app/examples
RUN pip install -r requirements.txt
RUN pip install -e ..
ENTRYPOINT python solve.py
