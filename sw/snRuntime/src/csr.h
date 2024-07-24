#ifndef CSR_H
#define CSR_H

uint32_t read_csr_soft_switch(uint32_t csr_address);
void write_csr_soft_switch(uint32_t csr_address, uint32_t value);

#endif  // CSR_H
