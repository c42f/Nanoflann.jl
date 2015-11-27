run(`git submodule update --init`)

if !isdir("build")
    mkdir("build")
end
cd("build") do
    run(`cmake ..`)
    run(`make`)
end
run(`cp build/deps.jl .`)
