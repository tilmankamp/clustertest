#!/usr/bin/bash

worker_count=1
worker_count=2
index=0

while [ "$index" -lt "$worker_count" ]
do
  rm -f worker_$index
  git clone `git remote get-url origin` worker_$index
  cd worker_$index
  sed -i -e 's/PARAMS/--job_name="worker" --task_index='$index'/g' riseml.yml
  git commit -m update
  riseml push
  cd ..
  echo "Started worker $index."
  ((index++))
done

while [ "$index" -lt "$ps_count" ]
do
  rm -f ps_$index
  git clone `git remote get-url origin` ps_$index
  cd ps_$index
  sed -i -e 's/PARAMS/--job_name="ps" --task_index='$index'/g' riseml.yml
  git commit -m update
  riseml push
  cd ..
  echo "Started parameter server $index."
  ((index++))
done
