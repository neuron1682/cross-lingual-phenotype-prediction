FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

WORKDIR /pvc
RUN apt-get update && apt-get install -y rsync htop tmux openssh-server ssh git && rm -rf /var/lib/apt/lists/* 
COPY resources/requirements.txt /tmp/ 
RUN python -m pip install --upgrade pip 
RUN pip  install -U -r /tmp/requirements.txt

RUN pip install -U  git+https://github.com/Adapter-Hub/adapter-transformers.git

RUN mkdir -p /var/run/sshd; \
    mkdir /root/.ssh && chmod 700 /root/.ssh; \
    touch /root/.ssh/authorized_keys

COPY sshd_config /etc/ssh/sshd_config
COPY src .

EXPOSE 22 
RUN export PYTHONPATH="${PYTHONPATH}:/pvc/"

CMD ["/usr/sbin/sshd", "-D"]