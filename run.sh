#!/bin/bash
python generate.py
cat test.out | python bhtsne.py -d 2 -p 0.1
