#!/bin/bash

for i in {1..11}; do
    echo "Target subject: $i"
    python main_SHU.py --target_sub "$i"
done
echo "fine!"