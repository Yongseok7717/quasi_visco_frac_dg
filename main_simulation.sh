#!/bin/bash

# Example 1: with user-defined solution
python run_uniform.py -g 20 -G 1 -k 1 -i 3 -I 8  
python run_uniform.py -g 20 -G 1 -k 2 -i 3 -I 8  

# Example 2: butyl rubber
python run_elasticity.py 
python run_viscoelasticity.py 
