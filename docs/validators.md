![ZANGIEF](images/zangief.png)

# Validator Documentation

## Hardware Requirements
 Minimum Requirements

    CPU: Quad-core Intel i3 or equivalent AMD processor, 2.5 GHz
    RAM: 2 GB
    Storage: 2 GB
    GPU: Not needed
    Network: Broadband internet connection for online data syncing

 Recommended Requirements

    CPU: 4-core Intel i5 or equivalent AMD processor, 2.5 GHz-3.5 GHz
    RAM: 4 GB or more
    Storage: 10 GB SSD
    GPU: Not needed
    Network: Gigabit Ethernet or better


## How to Run a Validator

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

`pip install -r validator_requirements.txt`

3) Register the validator

`comx module register <name> <your_commune_key> --netuid 1`

4) Set the environment variables

Copy the environment variable template and set the values in the `.env` file.

```bash
cp env/.example.env env/.env
```

**Using a `.env` file will override any environment variables already set.**

Alternatively, if you do not want to use a `.env` file you can set the environment variables directly and pass `--ignore-config` when running the validator.

5) Run the validator

```
python src/zangief/validator/validator.py
```

(Optional) Run with pm2 

```
sudo apt install jq -y && sudo apt install npm -y && sudo npm install pm2 -g && pm2 update
pm2 start --name zangief-vali "python src/zangief/validator/validator.py"
```

(Optional) Run on testnet

1) Register the validator on the testnet

`comx --testnet module register <name> <your_commune_key> --netuid 23`

2) Set the config.ini file to `isTestnet = 1`

`env/config.ini` 
```
[validator]
name = validator
keyfile = validator
interval = 600
isTestnet = 1
```

3) Run the validator

```
cd ~/zangief
python src/zangief/validator/validator.py
```