#!/bin/bash
bellgarland=("cant" "consph" "cop20k_A" "mac_econ_fwd500" "mc2depi" "pdb1HYS" "pwtk" "rma10" "scircuit" "shipsec1" "webbase-1M")
large=("Flan_1565" "delaunay_n24" "europe_osm")
small=("3elt" "Alemdar" "Dubcova1" "G51" "Na5" "Reuters911" "Si10H16" "add20" "celegansneural" "crack" "fxm3_6" "hydr1c" "oscil_dcop_11" "rajat04" "tomography")
spgemmdata=("2cubes_sphere" "cage12" "filter3D" "hood" "m133-b3" "majorbasis" "mario002" "mc2depi" "offshore" "patents_main" "poisson3Da" )

for value in "${bellgarland[@]}"; do
    $HOME/RCP/Output/bin/rcprof -p -o "$HOME/result/Bell_Garland/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Bell_Garland/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -O -o "$HOME/result/Bell_Garland/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Bell_Garland/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -t -o "$HOME/result/Bell_Garland/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Bell_Garland/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -T -a $HOME/result/Bell_Garland/${value}.atp
done
for value in "${large[@]}"; do
    $HOME/RCP/Output/bin/rcprof -p -o "$HOME/result/Large/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Large/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -O -o "$HOME/result/Large/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Large/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -t -o "$HOME/result/Large/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Large/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -T -a $HOME/result/Large/${value}.atp
done
for value in "${small[@]}"; do
    $HOME/RCP/Output/bin/rcprof -p -o "$HOME/result/Small/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Small/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -O -o "$HOME/result/Small/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Small/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -t -o "$HOME/result/Small/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/Small/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -T -a $HOME/result/Small/${value}.atp
done
for value in "${spgemmdata[@]}"; do
    $HOME/RCP/Output/bin/rcprof -p -o "$HOME/result/SpGemmData/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/SpGemmData/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -O -o "$HOME/result/SpGemmData/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/SpGemmData/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -t -o "$HOME/result/SpGemmData/${value}" "$HOME/clSPARSE/clSPARSE-build/staging/clsparse-bench" -d $HOME/clSPARSE/Externals/MTX/SpGemmData/${value} -f SpMSpM
    $HOME/RCP/Output/bin/rcprof -T -a $HOME/result/SpGemmData/${value}.atp
done