FROM python:3.10

WORKDIR /app

# Copy the project files
COPY shoe_splatter /app/shoe_splatter/
COPY checkpoints/ /app/checkpoints/
COPY pyproject.toml /app/pyproject.toml

# Install Python dependencies
RUN pip install -e .

CMD /bin/bash