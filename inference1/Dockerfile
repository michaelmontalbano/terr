FROM tensorflow/tensorflow:2.16.1-gpu

# Install minimal system dependencies
RUN apt-get update && \
    apt-get -y install sudo python3-pip python3-dev && \
    python3 -m pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy minimal requirements
COPY requirements.txt ./requirements.txt

# Install minimal Python requirements
RUN python3 -m pip install -r ./requirements.txt && \
    echo "Minimal packages installed successfully"

# Verify critical packages are installed
RUN python3 -c "import blosc2; print('blosc2 installed successfully')" && \
    python3 -c "import tensorflow; print('tensorflow installed successfully')" && \
    python3 -c "import boto3; print('boto3 installed successfully')" && \
    python3 -c "import redis; print('redis installed successfully')"

# Copy project code with all custom objects
COPY models.py ./models.py
COPY rnn.py ./rnn.py
COPY rnn_intervals.keras .
COPY model.h5 .
COPY config.py .
COPY predict.py .

# Verify custom objects can be imported
RUN python3 -c "from rnn import *; from models import *; print('Custom objects imported successfully')" && \
    echo "Container setup completed successfully"

# Set the default command to run predict.py
CMD ["python3", "predict.py"]
