FROM pytorch/pytorch

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 9000

CMD ["streamlit", "run","app.py"]