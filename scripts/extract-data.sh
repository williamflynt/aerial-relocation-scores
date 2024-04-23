#!/usr/bin/env sh

# Assumptions: you have the data and 7z and unzip.
# DEBIAN: sudo apt-get install -y p7zip-full

7z e aerialimagelabeling.7z.001 && \
unzip NEW2-AerialImageDataset.zip && \
rm NEW2-AerialImageDataset.zip
