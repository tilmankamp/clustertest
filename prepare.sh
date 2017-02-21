#!/bin/bash

ps_count=1
worker_count=2
prefix="exclu-1-"
count=((worker_count+ps_count))

index=0
while [ "$index" -lt "$count" ]
do
  name="$prefix$index"
  echo $name
  if [ ! -d "$name" ]; then
    git clone `git remote get-url origin` $name
    cd $name
    riseml create
    cd ..
  fi
  cd $name
  git fetch --all
  git rebase origin/master
  cp riseml.yml.template riseml.yml
  sed -i -e 's/PARAMS/--worker_count '$worker_count' --ps_count '$ps_count' --task_prefix '$prefix' --task_index='$index'/g' riseml.yml
  git add .
  git commit -m update
  riseml push $name
  cd ..
  echo "Started task $index."
  ((index++))
done
