FROM python:3.8

MAINTAINER Tomás Cabello López "tomascl1998@gmail.com"

ENV VIRTUAL_ENV=/opt/venv

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir /Concept-Drift

WORKDIR /Concept-Drift

COPY requirements.txt /Concept-Drift

RUN pip install -r requirements.txt

COPY . /Concept-Drift

ENTRYPOINT ["python"]

CMD ["run.py"]
