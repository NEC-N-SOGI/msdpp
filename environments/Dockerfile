FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG PROJECT_NAME=msdpp
ARG PROJECT_DIRECTORY=/${PROJECT_NAME}
ARG PYTHON_VERSION=3.10
ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
ARG LOCAL_USER_NAME=user

ENV LOCAL_UID=${LOCAL_UID} \
    LOCAL_GID=${LOCAL_GID} \
    LOCAL_USER_NAME=${LOCAL_USER_NAME} \
    HF_HOME=${PROJECT_DIRECTORY}/share_datasets/temp/

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${PROJECT_DIRECTORY}/src

ENV CUDA_HOME=/usr/local/cuda-11.8/

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update \
    && apt install --no-install-recommends -y \
        wget \
        zlib1g \
        ffmpeg \
        git \
        make \
        vim \
        gcc \
        tmux \
        default-jre \
        libpq-dev \
        zip \
        unzip \
        ca-certificates \
        curl \
    && echo "Y" | apt install sudo -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u ${LOCAL_UID} -o -m ${LOCAL_USER_NAME} 2> /dev/null && \
    groupmod -g ${LOCAL_GID} ${LOCAL_USER_NAME}

RUN usermod -aG sudo ${LOCAL_USER_NAME} \
    && echo "${LOCAL_USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/trusted-user

RUN mkdir chown /home/${LOCAL_USER_NAME}/.cache && \
    chown -R ${LOCAL_USER_NAME}:${LOCAL_USER_NAME} /home/${LOCAL_USER_NAME}/.cache


USER ${LOCAL_USER_NAME}
WORKDIR ${PROJECT_DIRECTORY}

# Install the project's dependencies using the lockfile and settings
# RUN --mount=type=cache,target=/home/${USER_NAME}/.cache/uv,--readwrite \
COPY --chown=${LOCAL_USER_NAME}:${LOCAL_USER_NAME} pyproject.toml ${PROJECT_DIRECTORY}/pyproject.toml
COPY --chown=${LOCAL_USER_NAME}:${LOCAL_USER_NAME} uv.lock ${PROJECT_DIRECTORY}/uv.lock

RUN uv venv -p ${PYTHON_VERSION} \
    && uv sync --frozen --all-extras --native-tls

RUN cd ~/ \
    && wget https://panderson.me/images/SPICE-1.0.zip \
    && unzip SPICE-1.0.zip

