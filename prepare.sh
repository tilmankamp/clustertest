#!/bin/bash

ps_count=1
worker_count=2
prefix="_t6_"

index=0
while [ "$index" -lt "$worker_count" ]
do
  name="worker$prefix$index"
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
  sed -i -e 's/PARAMS/--job_name="worker" --task_index='$index'/g' riseml.yml
  git add .
  git commit -m update
  riseml push
  cd ..
  echo "Started worker $index."
  ((index++))
done

index=0
while [ "$index" -lt "$ps_count" ]
do
  name="ps$prefix$index"
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
  sed -i -e 's/PARAMS/--job_name="ps" --task_index='$index'/g' riseml.yml
  git add .
  git commit -m update
  riseml push
  cd ..
  echo "Started parameter server $index."
  ((index++))
done
