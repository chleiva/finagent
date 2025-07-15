#!/bin/bash

TARGET=600

while true; do
  count=$(ls *.csv 2>/dev/null | wc -l)
  clear
  echo "🗂️  CSV File Progress: $count / $TARGET"
  
  # Optional: progress bar
  filled=$(printf "%0.s█" $(seq 1 $((count * 50 / TARGET))))
  empty=$(printf "%0.s " $(seq 1 $((50 - count * 50 / TARGET))))
  echo "[$filled$empty]"

  if [ "$count" -ge "$TARGET" ]; then
    echo "✅ Target reached!"
    break
  fi

  sleep 1
done

