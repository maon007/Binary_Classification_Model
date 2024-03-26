FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and data file to the container
COPY data_exploration.py .
COPY data_processing.py .
COPY data_modeling.py .
COPY main.py .
COPY it_data.csv .

# Command to run the Python script with the data file as an argument
CMD ["python", "main.py", "data_1.csv"]
