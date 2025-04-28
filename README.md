# featureSelection

Cloning the repository:
```sh
git clone https://github.com/tobiasfremming/featureSelection.git
```

### Generating the LUTs
```sh
git submodule init
git submodule update
cd training
python [file]
```

### NSGA
```sh
cd src
julia --project=. nsga.jl [dataset-name] # dataset name is heart, diabetes, cancer or wine
```

### SWARM
```sh
cd src
cd swarm
julia --project=. bpso.jl [dataset-name] # dataset name is heart, diabetes, cancer or wine
```

Or simply navigate to src/swarm/bpso.jl and run

### Task 6
```sh
cd src
julia --project=. sga_blind.jl [dataset-name] # dataset name is cleveland, zoo or letter
```

