#include <stdint.h>

const int CONFIG_SIZE_LUT = 16 * 3;
const int CONFIG_SIZE_DATA = 16 * 1;
const int CONFIG_SIZE_CMD_0 = 16 * 1 * 2;
const int CONFIG_SIZE_CMD_SS = 16 * (1 + 1) * 2;

const int delta_config_base = 1408;
const int delta_comp_data = 2688;
const int delta_store_data = 9152;

const int delta_config_lut = delta_config_base;
const int delta_config_data = delta_config_lut + CONFIG_SIZE_LUT * 4;
const int delta_config_cmd_0 = delta_config_data + CONFIG_SIZE_DATA * 4;
const int delta_config_cmd_ss = delta_config_cmd_0 + CONFIG_SIZE_CMD_0 * 4;

uint32_t CONFIG_CMD[CONFIG_SIZE_CMD_0] = {
	// Program begin
	1817185216,  1853885376,  1853885376,  1652952000,  2483555264,  2516585408,  2516585408,  1095764928,  
	1213205440,  1853885376,  1853885376,  1652952000,  2483030976,  2516585408,  2516585408,  1095764928,  
	1510477828,  1510477828,  1510477828,  1510477825,  1510477825,  1510477827,  1510477827,  1510477827,  
	1510477828,  1510477828,  1510477828,  1510477825,  1510477827,  1510477827,  1510477827,  1510477827,  
	// Program end
};

uint32_t CONFIG_CMD_SS[CONFIG_SIZE_CMD_SS] = {
	// Program begin
	1817185216,  1853885376,  1853885376,  1652952000,  2483555264,  2516585408,  2516585408,  1095764928,  
	1213205440,  1853885376,  1853885376,  1652952000,  2483030976,  2516585408,  2516585408,  1095764928,  
	1510477828,  1510477828,  1510477828,  1510477825,  1510477825,  1510477827,  1510477827,  1510477827,  
	1510477828,  1510477828,  1510477828,  1510477825,  1510477827,  1510477827,  1510477827,  1510477827,  

	         0,           0,           0,           0,           0,           0,           0,           0,  
	         0,           0,           0,           0,           0,           0,           0,           0,  
	         0,           0,           0,           0,           0,           0,           0,           0,  
	         0,           0,           0,           0,           0,           0,           0,           0,  
	// Program end
};
