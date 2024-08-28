# Use the official Python image as a base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Expose the port your application will run on
EXPOSE 8080

# Command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
