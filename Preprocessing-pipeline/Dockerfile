FROM python:3.10.9-alpine
WORKDIR /app
RUN apk add --no-cache gfortran build-base gcc ffmpeg
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py", "/data"]