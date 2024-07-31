# Testing to get it working

1. Include files `tm_model.c`, `tm_layers.c`, `tm_stat.c`, `tinymaix.h`, `tm_port.h`, `arch_cpu.h` in this way:

```
program_folder\lib
└───tinymaix
    ├───include
    │       tinymaix.h
    │       tm_port.h
    │
    └───src
            arch_cpu.h
            tm_layers.c
            tm_model.c
            tm_stat.c
```

2. Orchestrate file structure for using both python and computer build, aswell as gd32 build
   - Python needs include and src in root folder
   - MCU needs them in lib/src and lib/include  

3. Install neccesary python libaries from `requirments.txt` in root folder.