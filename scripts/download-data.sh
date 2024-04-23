#! /usr/bin/env sh

# Download data from last known good link.

mkdir "./data"

curl --request GET -sL \
     --url 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001'\
     --output './data/aerialimagelabeling.7z.001'

curl --request GET -sL \
     --url 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002'\
     --output './data/aerialimagelabeling.7z.001'

curl --request GET -sL \
     --url 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003'\
     --output './data/aerialimagelabeling.7z.001'

curl --request GET -sL \
     --url 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004'\
     --output './data/aerialimagelabeling.7z.001'

curl --request GET -sL \
     --url 'https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005'\
     --output './data/aerialimagelabeling.7z.001'
