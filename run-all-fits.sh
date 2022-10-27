#!/bin/bash

for fit_type in `echo "ml bayesian"`; do
    for model in `echo "2.3 3.3 3.4 4.4 4.5"`; do
        for genotype in `echo "UxxGx Uxxxx xCxGx xCxxx xxAGx xxAxx xxxGx xxxxx"`; do
            ./do_fit.py ${genotype} ${fit_type} ${model}
            exit
        done
    done
done

