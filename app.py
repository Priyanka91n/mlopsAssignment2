dockerfile_code = '''
# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
'''

with open("Dockerfile", "w") as f:
    f.write(dockerfile_code.strip())

print("✅ Dockerfile created successfully.")