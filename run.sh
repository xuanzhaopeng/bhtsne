#!/bin/bash
rm in.out
rm result.out
python generate.py
cat in.out | python bhtsne.py -d 2 -p 0.1 -o result.out
