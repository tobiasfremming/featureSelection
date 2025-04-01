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

### SGA
```sh
cd src
julia --project=. sga.jl [dataset-name] # dataset name is heart, diabetes, cancer or wine
```
