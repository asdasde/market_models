FROM python:3.12
EXPOSE 8081

RUN groupadd -g 1000 xcentric && useradd -u 1000 -g xcentric -m -s /bin/bash xcentric

COPY --chown=xcentric:xcentric ./ /home/xcentric/app

USER xcentric
WORKDIR /home/xcentric/app

RUN pip install --root-user-action=ignore -r requirements.txt 
RUN pip install --root-user-action=ignore pyarrow

WORKDIR /home/xcentric/app/src
CMD python api/main.py

