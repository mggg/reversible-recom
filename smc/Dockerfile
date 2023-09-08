FROM r-base:4.3.0
RUN apt update
RUN apt install -y python3 python3-pip
RUN apt install -y libcurl4-openssl-dev libudunits2-dev libssl-dev libjq-dev libopenmpi-dev libgdal-dev
RUN Rscript -e 'install.packages("remotes")'
RUN Rscript -e 'remotes::install_github("alarm-redist/redist",  dependencies=TRUE)'
RUN Rscript -e 'install.packages(c("rgdal", "broom", "IRkernel"))'

RUN apt install -y python3-dev 
RUN pip3 install --break-system-packages matplotlib pandas jupyterlab==3.3.1 rpy2
RUN Rscript -e 'IRkernel::installspec()'

# New dependencies for fifty-states data processing.
# (Not required for just running SMC.)
RUN Rscript -e 'remotes::install_github("christopherkenny/geomander", dependencies=TRUE)'
RUN apt install -y libbz2-dev liblzma-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libgit2-dev
RUN Rscript -e 'install.packages(c("here", "devtools"), dependencies=TRUE)'
RUN Rscript -e 'install.packages(c("cvap", "tigris", "PL94171"), dependencies=TRUE)'


#CMD ["jupyter", "lab", "--allow-root", "--no-browser", "--ip", "0.0.0.0", "/app"]
CMD ["bash"]
