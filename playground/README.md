# BarreCUDA/playground
- `add.cu` is a simple program to perform addition of two vectors.
- Specify the `-arch` flag to avoid compiling for older architectures.
    ```sh
    nvcc -arch=sm_86 add.cu -o add_vec
    ./add_vec
    ```
