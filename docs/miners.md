![ZANGIEF](images/zangief.png)


# Miner Documentation

## Miners
* [OpenAI Miner](#How-To-run-the-openai-miner)
* [M2M_100 Miner](#How-To-run-the-m2m_100-miner)


## How To run the OpenAI miner:

### Hardware Requirements

 Recommended Requirements

    CPU: 4-core Intel i5 or equivalent AMD processor, 2.5 GHz-3.5 GHz
    RAM: 4 GB or more
    Storage: 8 GB SSD
    GPU: Not needed
    Network: Gigabit Ethernet or better


> [!NOTE]
> Requires python3.10


1) Clone project

`git clone https://github.com/nakamoto-ai/zangief`


2) Create virtual environment

```
cd zangief
python -m venv venv
source venv/bin/activate
```

3) Install dependencies

`pip install -r openai_miner_requirements.txt`

4) Add OpenAI API key and OpenAI model to the `env/config.ini` file

`env/config.ini` (sample)
```
[miner]
keyfile = miner
url = http://0.0.0.0:5000/
isTestnet = 0
openai_key = YOUR_OPENAI_KEY
model = gpt-3.5-turbo
```

5) Register the miner

`comx module register <name> <your_commune_key> --netuid 1 --ip <your_ip> --port <your_port>`

6) Run the miner

```
python src/zangief/miner/miner.py --miner openai
```

(Optional) Run with pm2 

```
sudo apt install jq -y && sudo apt install npm -y && sudo npm install pm2 -g && pm2 update
pm2 start --name zangief-openai "python src/zangief/miner/openai_miner.py"
```

## How To run the M2M_100 miner:

### Hardware Requirements


Recommended Requirements

    CPU: 4-core Intel i5 or equivalent AMD processor, 2.5 GHz-3.5 GHz
    RAM: 8 GB
    Storage: 16 GB
    GPU: RTX A4000 or better
    Network: Gigabit Ethernet or better


> [!NOTE]
> Requires python3.10


1) Clone project

`git clone https://github.com/nakamoto-ai/zangief`

2) Create virtual environment

```
cd zangief
python -m venv venv
source venv/bin/activate
```

3) Install dependencies

`pip install -r m2m_miner_requirements.txt`

4) Register the miner

`comx module register <name> <your_commune_key> --netuid 1 --ip <your_ip> --port <your_port>`

5) Run the miner

```
python src/zangief/miner/miner.py --miner m2m
```

(Optional) Run with pm2 

```
sudo apt install jq -y && sudo apt install npm -y && sudo npm install pm2 -g && pm2 update
pm2 start --name zangief-m2m_100 "python src/zangief/miner/m2m_miner.py"
```
