#!/bin/bash

echo -e "\e[31mINSTALLING E6111 VM SETUP SOFTWARE\e[0m"
sudo apt-get -y update
sudo apt-get install python3-pip
python3 -m pip install pip --upgrade
sudo apt-get install python3.7
sudo apt-get install python3.7-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo pip3 install --upgrade google-api-python-client

echo -e "\e[31mINSTALLING PROJECT SPECIFIC SOFTWARE\e[0m"

sudo apt update
sudo apt install python3-pip
pip3 install beautifulsoup4
sudo apt-get update
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_lg 
pip3 install -r requirements.txt
bash download_finetuned.sh
pip3 install openai
