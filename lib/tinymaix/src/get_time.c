#include"gd32vf103.h"
#include "stdio.h"
// Function to read the cycle counter
uint64_t get_cycle_value_asm(void) {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}


uint32_t get_time(void) {
    uint64_t cycles = get_cycle_value_asm();  
    return (uint32_t)((cycles * 1000000) / SystemCoreClock);
}