#!/bin/bash

python run_uniform.py -g 20 -G 1 -k 1 -i 3 -I 8  2>&1 | tee -i ex1_rate_linear.txt
python run_uniform.py -g 20 -G 1 -k 2 -i 3 -I 8  2>&1 | tee -i ex1_rate_quad.txt