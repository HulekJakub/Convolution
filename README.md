### How to run

1. Download all of the files and unpack them.
2. Enter the folder.
3. a) run from the base directory    
    ``` 
    mkdir build
    cd build/
    cmake .. 
    make
    ./my-convolution
    ```
   b) with profiling    
    install **FlameGraph** and modify the "run_profiling" script to point to your installtion
    the run from the base directory
    ```
    ./run_profiling
    ```