#!/bin/bash

ps_count=1
worker_count=2
prefix="exclu-3-"
count=$((worker_count + ps_count))

index=0
while [ "$index" -lt "$count" ]
do
  name="$prefix$index"
  echo $name
  if [ ! -d "$name" ]; then
    mkdir $name
    cd $name
    git init
    riseml create
    cd ..
  fi
  cd $name
  cp -r ../node/* .
  sed -i -e 's/PARAMS/--worker_count '$worker_count' --ps_count '$ps_count' --task_prefix '$prefix' --task_index='$index'/g' riseml.yml
  git add .
  git commit -m update
  riseml push $name 2>&1 | sed "s/^/[task "$index"] /" &
  cd ..
  echo "Started task $index."
  ((index++))
done
