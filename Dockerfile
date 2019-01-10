# Environment for running, testing and releasing hmm_filter.

# Start from Alpine + Python 3.6
FROM frolvlad/alpine-python3

# Update package index
RUN apk add --update

# Install build environment and libraries required to install pip packages
RUN apk add bash bash-completion make vim g++ gcc python3-dev musl-dev libffi-dev openssl-dev openblas-dev freetype-dev

# Upgrade pip and install packages required for testing and packaging
RUN pip install --upgrade pip

# required, otherwise installation of requirements.txt fails
RUN pip install cffi

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# install machine learning packages in specific order to avoid compliation or import issues
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install matplotlib

# Set default working directory
WORKDIR /code

# Run Jupiter
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token=", "./notebooks"]
