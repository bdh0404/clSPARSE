#!/bin/bash
small=("3elt" "Alemdar" "Dubcova1" "G51" "Na5" "Reuters911" "Si10H16" "add20" "celegansneural" "crack" "fxm3_6" "hydr1c" "rajat04" "tomography")

for value in "${small[@]}"; do
 ./clSPARSE-build/staging/test-blas3 -p "./Externals/MTX/Small/${value}/${value}.mtx" -f SpMSpM
done