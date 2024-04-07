FROM python:3.10

# set the working directory
WORKDIR /app

# install dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy code to the folder
COPY . /app

# install the library
CMD pip install . && python