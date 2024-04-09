# Use the official Python image as the base
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Metaflow and any other required packages
RUN pip install metaflow boto3

# Set AWS credentials and Metaflow configuration as environment variables
# Replace YOUR_BUCKET_NAME with your actual S3 bucket name
ENV AWS_ACCESS_KEY_ID=your_access_key_id
ENV AWS_SECRET_ACCESS_KEY=your_secret_access_key
ENV AWS_DEFAULT_REGION=your_aws_region
ENV METAFLOW_DATATOOLS_S3ROOT=s3://YOUR_BUCKET_NAME/metaflow/
ENV METAFLOW_DATASTORE_SYSROOT_S3=s3://YOUR_BUCKET_NAME/metaflow/

# Run the Metaflow script when the container launches
CMD ["python", "sample_flow.py"]
