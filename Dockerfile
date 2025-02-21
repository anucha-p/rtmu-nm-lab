# app/Dockerfile

# FROM ideonate/streamlit-single:latest
FROM python:3.9-slim
WORKDIR /app
# RUN chsh -s /bin/bash && apt-get update && apt-get install curl -y
# RUN apt-get install -y build-essential 
# RUN apt-get install -y curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*


# RUN git clone https://github.com/streamlit/streamlit-example.git .
# RUN apt-get update && apt-get install -y freeglut3-dev \
#     libgtk2.0-dev 


COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py","--server.port=8501", "--server.address=0.0.0.0"]