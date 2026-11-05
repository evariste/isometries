#!/bin/bash



for f in *.py
do

  echo "Running $f"
  uv run "${f}"
  echo
  echo "================================================================"
  echo
done

