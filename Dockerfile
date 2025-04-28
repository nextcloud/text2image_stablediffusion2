FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

COPY requirements.txt /

ENV DEBIAN_FRONTEND=noninteractive

RUN \
   apt-get update -y && \
   apt-get install -y software-properties-common && \
   add-apt-repository -y ppa:deadsnakes/ppa && \
   apt-get update -y && \
   apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip vim git pciutils && \
   update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN \
  python3 -m pip install -r requirements.txt && rm -rf ~/.cache && rm requirements.txt

ADD /ex_app/cs[s] /ex_app/css
ADD /ex_app/im[g] /ex_app/img
ADD /ex_app/j[s] /ex_app/js
ADD /ex_app/l10[n] /ex_app/l10n
ADD /ex_app/li[b] /ex_app/lib

COPY --chmod=775 healthcheck.sh /

WORKDIR /ex_app/lib
ENTRYPOINT ["python3", "main.py"]
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /healthcheck.sh
