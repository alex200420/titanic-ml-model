# Use an official Python runtime as a parent image
FROM python:3.9-slim as build

# Add metadata to an image
LABEL maintainer="alejandrob.jimenezp@gmail.com"
LABEL version="0.1"
LABEL description="Docker image for setting up Titanic ML Repo"

RUN mkdir titanic_ml
COPY . titanic_ml/
WORKDIR titanic_ml/

# Install package
RUN pip install .