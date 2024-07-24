// Soft switch for CSR to make it support dynamic addressing
// The function can address 32 CSR registers starting from 960

#include "stdint.h"

uint32_t read_csr_soft_switch(uint32_t csr_address) {
    uint32_t value;
    switch (csr_address) {
        case 960:
            return read_csr(960);
        case 961:
            return read_csr(961);
        case 962:
            return read_csr(962);
        case 963:
            return read_csr(963);
        case 964:
            return read_csr(964);
        case 965:
            return read_csr(965);
        case 966:
            return read_csr(966);
        case 967:
            return read_csr(967);
        case 968:
            return read_csr(968);
        case 969:
            return read_csr(969);
        case 970:
            return read_csr(970);
            break;
        case 971:
            return read_csr(971);
        case 972:
            return read_csr(972);
        case 973:
            return read_csr(973);
        case 974:
            return read_csr(974);
            break;
        case 975:
            return read_csr(975);
        case 976:
            return read_csr(976);
        case 977:
            return read_csr(977);
        case 978:
            return read_csr(978);
        case 979:
            return read_csr(979);
        case 980:
            return read_csr(980);
        case 981:
            return read_csr(981);
        case 982:
            return read_csr(982);
        case 983:
            return read_csr(983);
        case 984:
            return read_csr(984);
        case 985:
            return read_csr(985);
        case 986:
            return read_csr(986);
        case 987:
            return read_csr(987);
            break;
        case 988:
            return read_csr(988);
        case 989:
            return read_csr(989);
        case 990:
            return read_csr(990);
        case 991:
            return read_csr(991);
    }
    return 0;
}

void write_csr_soft_switch(uint32_t csr_address, uint32_t value) {
    switch (csr_address) {
        case 960:
            write_csr(960, value);
            break;
        case 961:
            write_csr(961, value);
            break;
        case 962:
            write_csr(962, value);
            break;
        case 963:
            write_csr(963, value);
            break;
        case 964:
            write_csr(964, value);
            break;
        case 965:
            write_csr(965, value);
            break;
        case 966:
            write_csr(966, value);
            break;
        case 967:
            write_csr(967, value);
            break;
        case 968:
            write_csr(968, value);
            break;
        case 969:
            write_csr(969, value);
            break;
        case 970:
            write_csr(970, value);
            break;
        case 971:
            write_csr(971, value);
            break;
        case 972:
            write_csr(972, value);
            break;
        case 973:
            write_csr(973, value);
            break;
        case 974:
            write_csr(974, value);
            break;
        case 975:
            write_csr(975, value);
            break;
        case 976:
            write_csr(976, value);
            break;
        case 977:
            write_csr(977, value);
            break;
        case 978:
            write_csr(978, value);
            break;
        case 979:
            write_csr(979, value);
            break;
        case 980:
            write_csr(980, value);
            break;
        case 981:
            write_csr(981, value);
            break;
        case 982:
            write_csr(982, value);
            break;
        case 983:
            write_csr(983, value);
            break;
        case 984:
            write_csr(984, value);
            break;
        case 985:
            write_csr(985, value);
            break;
        case 986:
            write_csr(986, value);
            break;
        case 987:
            write_csr(987, value);
            break;
        case 988:
            write_csr(988, value);
            break;
        case 989:
            write_csr(989, value);
            break;
        case 990:
            write_csr(990, value);
            break;
        case 991:
            write_csr(991, value);
            break;
    }
}
