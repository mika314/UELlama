# Llama.cpp Build Parameters

Llama.cpp was built from git hash: `dadbed99e65252d79f81101a392d0d6497b86caa`

With the following build commands:

```
mkdir build
cd build/
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_CUDA_DMMV_X=64 -DLLAMA_CUDA_MMV_Y=2 -DLLAMA_CUDA_F16=true -DBUILD_SHARED_LIBS=ON
cd ..
cmake --build build --config Release -j --verbose
```

Then the .so or .lib file was copied into the `Libraries` directory and all the .h files were copied to the `Includes` directory. In Windows you should put the build/bin/llama.dll into `Binaries/Win64` directory.
