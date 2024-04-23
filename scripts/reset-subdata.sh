#! /usr/bin/env sh

cd data && rm -rf subdata
mkdir "subdata" && touch ./subdata/.gitkeep && git add -f ./subdata/.gitkeep
