# Nanoflann

Very basic C function wrappers around the
[nanoflann](https://github.com/jlblancoc/nanoflann) k nearest neighbors library,
and associated julia interface on top of that.  This was created mainly for a
benchmark comparison with NearestNeighbors.jl; it probably won't go any further.

Instructions: Compile src/nanoflann.cpp using a C++ compiler and run
test/benchmark.jl from the test directory.

On linux, you'll need something like

```
g++ -O3 -shared -fPIC -o nanoflann.so nanoflann.cpp -I/path/to/nanoflann-1.1.8/
```

