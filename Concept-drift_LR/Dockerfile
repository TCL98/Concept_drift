FROM python:3.8

RUN mkdir /Concept-Drift

WORKDIR /Concept-Drift

COPY requirements.txt /Concept-Drift

RUN pip install -r requirements.txt

COPY . /Concept-Drift

CMD [ "python", "./run.py" ]
