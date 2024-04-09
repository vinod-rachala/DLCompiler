# Use the official Python image as the base
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install Metaflow
RUN pip install metaflow

# Copy the Metaflow script and the Metaflow config file into the container
COPY sample_flow.py config.json ./

# Metaflow will automatically look for a config.json file in the current directory,
# the user's home directory, or /metaflow.
# Set the HOME environment variable so Metaflow finds config.json in /app.
ENV HOME=/app

# Run the Metaflow script
CMD ["python", "sample_flow.py"]
