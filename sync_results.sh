#!/bin/bash
cd /home/stefan/git/gsnoop_experiment/
git pull
git add results/*
git commit -am 'autocommit'
git push
