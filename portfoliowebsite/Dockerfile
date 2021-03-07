FROM python:3.8-buster

RUN pip install django==3.1 gunicorn==20.0 djangorestframework
RUN pip install pandas scikit-learn==0.23
RUN pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch_lightning
COPY . /var/www

WORKDIR /var/www
