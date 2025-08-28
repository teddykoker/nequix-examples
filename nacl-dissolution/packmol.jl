import Pkg; Pkg.activate("Packmol", shared=true)
using Packmol
for input_file in ARGS
    run_packmol(input_file)
end
