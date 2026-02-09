FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    iputils-ping curl wget git unzip zip zsh vim gnupg tree \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "rembg[cpu]"

# Silence onnxruntime
ENV TF_CPP_MIN_LOG_LEVEL=3

# Install oh-my-zsh and set up as default shell
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    chsh -s $(which zsh)

CMD ["tail", "-f", "/dev/null"]