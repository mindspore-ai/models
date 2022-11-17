#!/bin/bash

python3.7 main.py --config=../data/model/DCGAN/generator_config.json \
                  --nimages=1 \
                  --save_path=../data/sdk_result/
