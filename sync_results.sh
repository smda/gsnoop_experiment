#!/bin/bash
cd /home/muehlbauer/git/gsnoop_experiment/
git pull
git add results/*
git commit -am 'autocommit'
git push
