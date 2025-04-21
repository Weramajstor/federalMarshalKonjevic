generiranje.exe
del solution.log
scip -s params.set -f set_cover_problem.lp -l solution.log
solution.log
