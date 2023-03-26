#!/bin/bash

echo -e "\e[31mINSTALLING E6111 VM SETUP SOFTWARE\e[0m"

# Update package lists and install required software
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y python3-pip python3.7 python3.7-dev

# Upgrade pip and install google-api-python-client
python3 -m pip install --upgrade pip
pip3 install --upgrade google-api-python-client

echo -e "\e[31mINSTALLING PROJECT SPECIFIC SOFTWARE\e[0m"

# Install beautifulsoup4, spacy, and spacy's en_core_web_lg
pip3 install beautifulsoup4
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_lg

# Install packages from requirements.txt, download finetuned model, and install openai
pip3 install -r requirements.txt
bash download_finetuned.sh
pip3 install openai
