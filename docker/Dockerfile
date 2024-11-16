FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt update && apt install -y curl python3.10 python3-pip

COPY pyproject.toml .
ENV PATH="${PATH}:/root/.local/bin"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry install

CMD ["/bin/bash"]
