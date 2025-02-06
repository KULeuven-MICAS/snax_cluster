// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>

// This file provides the function to read and write CSR with CSR address in
// register As CSR instruction in RISC-V is immediate-number addressed, this
// workaround function deploys a switch-case to map the CSR address to implement
// pseudo register-mapping mechanism. To avoid the loss of performance, the
// function is defined in header, so that it can be compiled together with the
// main program, and optimized by the compiler. If @csr_address is provided in
// an immediate number, (macros, constant etc.) the compiler won't add it in a
// separate function creating loss in switching cycles.

#ifndef CSR_H
#define CSR_H
#define CSR_LONG_ADDR_MODE
// Uncomment the above line to enable 64 CSRs addressability, with the down side
// of larger binary size.

static void write_csr_obs(uint32_t value) {
    write_csr(1989, value);
    return;
}

static uint32_t read_csr_obs(void) { return read_csr(1989); }

static uint32_t csrr_ss(uint32_t csr_address) {
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
        case 971:
            return read_csr(971);
        case 972:
            return read_csr(972);
        case 973:
            return read_csr(973);
        case 974:
            return read_csr(974);
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
        case 988:
            return read_csr(988);
        case 989:
            return read_csr(989);
        case 990:
            return read_csr(990);
        case 991:
            return read_csr(991);
#ifdef CSR_LONG_ADDR_MODE
        case 992:
            return read_csr(992);
        case 993:
            return read_csr(993);
        case 994:
            return read_csr(994);
        case 995:
            return read_csr(995);
        case 996:
            return read_csr(996);
        case 997:
            return read_csr(997);
        case 998:
            return read_csr(998);
        case 999:
            return read_csr(999);
        case 1000:
            return read_csr(1000);
        case 1001:
            return read_csr(1001);
        case 1002:
            return read_csr(1002);
        case 1003:
            return read_csr(1003);
        case 1004:
            return read_csr(1004);
        case 1005:
            return read_csr(1005);
        case 1006:
            return read_csr(1006);
        case 1007:
            return read_csr(1007);
        case 1008:
            return read_csr(1008);
        case 1009:
            return read_csr(1009);
        case 1010:
            return read_csr(1010);
        case 1011:
            return read_csr(1011);
        case 1012:
            return read_csr(1012);
        case 1013:
            return read_csr(1013);
        case 1014:
            return read_csr(1014);
        case 1015:
            return read_csr(1015);
        case 1016:
            return read_csr(1016);
        case 1017:
            return read_csr(1017);
        case 1018:
            return read_csr(1018);
        case 1019:
            return read_csr(1019);
        case 1020:
            return read_csr(1020);
        case 1021:
            return read_csr(1021);
        case 1022:
            return read_csr(1022);
        case 1023:
            return read_csr(1023);
        case 1024:
            return read_csr(1024);
        case 1025:
            return read_csr(1025);
        case 1026:
            return read_csr(1026);
        case 1027:
            return read_csr(1027);
        case 1028:
            return read_csr(1028);
        case 1029:
            return read_csr(1029);
        case 1030:
            return read_csr(1030);
        case 1031:
            return read_csr(1031);
        case 1032:
            return read_csr(1032);
        case 1033:
            return read_csr(1033);
        case 1034:
            return read_csr(1034);
        case 1035:
            return read_csr(1035);
        case 1036:
            return read_csr(1036);
        case 1037:
            return read_csr(1037);
        case 1038:
            return read_csr(1038);
        case 1039:
            return read_csr(1039);
        case 1040:
            return read_csr(1040);
        case 1041:
            return read_csr(1041);
        case 1042:
            return read_csr(1042);
        case 1043:
            return read_csr(1043);
        case 1044:
            return read_csr(1044);
        case 1045:
            return read_csr(1045);
        case 1046:
            return read_csr(1046);
        case 1047:
            return read_csr(1047);
        case 1048:
            return read_csr(1048);
        case 1049:
            return read_csr(1049);
        case 1050:
            return read_csr(1050);
        case 1051:
            return read_csr(1051);
        case 1052:
            return read_csr(1052);
        case 1053:
            return read_csr(1053);
        case 1054:
            return read_csr(1054);
        case 1055:
            return read_csr(1055);
        case 1056:
            return read_csr(1056);
        case 1057:
            return read_csr(1057);
        case 1058:
            return read_csr(1058);
        case 1059:
            return read_csr(1059);
        case 1060:
            return read_csr(1060);
        case 1061:
            return read_csr(1061);
        case 1062:
            return read_csr(1062);
        case 1063:
            return read_csr(1063);
        case 1064:
            return read_csr(1064);
        case 1065:
            return read_csr(1065);
        case 1066:
            return read_csr(1066);
        case 1067:
            return read_csr(1067);
        case 1068:
            return read_csr(1068);
        case 1069:
            return read_csr(1069);
        case 1070:
            return read_csr(1070);
        case 1071:
            return read_csr(1071);
        case 1072:
            return read_csr(1072);
        case 1073:
            return read_csr(1073);
        case 1074:
            return read_csr(1074);
        case 1075:
            return read_csr(1075);
        case 1076:
            return read_csr(1076);
        case 1077:
            return read_csr(1077);
        case 1078:
            return read_csr(1078);
        case 1079:
            return read_csr(1079);
        case 1080:
            return read_csr(1080);
        case 1081:
            return read_csr(1081);
        case 1082:
            return read_csr(1082);
        case 1083:
            return read_csr(1083);
        case 1084:
            return read_csr(1084);
        case 1085:
            return read_csr(1085);
        case 1086:
            return read_csr(1086);
        case 1087:
            return read_csr(1087);
        case 1088:
            return read_csr(1088);
        case 1089:
            return read_csr(1089);
        case 1090:
            return read_csr(1090);
        case 1091:
            return read_csr(1091);
        case 1092:
            return read_csr(1092);
        case 1093:
            return read_csr(1093);
        case 1094:
            return read_csr(1094);
        case 1095:
            return read_csr(1095);
        case 1096:
            return read_csr(1096);
        case 1097:
            return read_csr(1097);
        case 1098:
            return read_csr(1098);
        case 1099:
            return read_csr(1099);
        case 1100:
            return read_csr(1100);
        case 1101:
            return read_csr(1101);
        case 1102:
            return read_csr(1102);
        case 1103:
            return read_csr(1103);
        case 1104:
            return read_csr(1104);
        case 1105:
            return read_csr(1105);
        case 1106:
            return read_csr(1106);
        case 1107:
            return read_csr(1107);
        case 1108:
            return read_csr(1108);
        case 1109:
            return read_csr(1109);
        case 1110:
            return read_csr(1110);
        case 1111:
            return read_csr(1111);
        case 1112:
            return read_csr(1112);
        case 1113:
            return read_csr(1113);
        case 1114:
            return read_csr(1114);
        case 1115:
            return read_csr(1115);
        case 1116:
            return read_csr(1116);
        case 1117:
            return read_csr(1117);
        case 1118:
            return read_csr(1118);
        case 1119:
            return read_csr(1119);
        case 1120:
            return read_csr(1120);
        case 1121:
            return read_csr(1121);
        case 1122:
            return read_csr(1122);
        case 1123:
            return read_csr(1123);
        case 1124:
            return read_csr(1124);
        case 1125:
            return read_csr(1125);
        case 1126:
            return read_csr(1126);
        case 1127:
            return read_csr(1127);
        case 1128:
            return read_csr(1128);
        case 1129:
            return read_csr(1129);
        case 1130:
            return read_csr(1130);
        case 1131:
            return read_csr(1131);
        case 1132:
            return read_csr(1132);
        case 1133:
            return read_csr(1133);
        case 1134:
            return read_csr(1134);
        case 1135:
            return read_csr(1135);
        case 1136:
            return read_csr(1136);
        case 1137:
            return read_csr(1137);
        case 1138:
            return read_csr(1138);
        case 1139:
            return read_csr(1139);
        case 1140:
            return read_csr(1140);
        case 1141:
            return read_csr(1141);
        case 1142:
            return read_csr(1142);
        case 1143:
            return read_csr(1143);
        case 1144:
            return read_csr(1144);
        case 1145:
            return read_csr(1145);
        case 1146:
            return read_csr(1146);
        case 1147:
            return read_csr(1147);
        case 1148:
            return read_csr(1148);
        case 1149:
            return read_csr(1149);
        case 1150:
            return read_csr(1150);
        case 1151:
            return read_csr(1151);
        case 1152:
            return read_csr(1152);
        case 1153:
            return read_csr(1153);
        case 1154:
            return read_csr(1154);
        case 1155:
            return read_csr(1155);
        case 1156:
            return read_csr(1156);
        case 1157:
            return read_csr(1157);
        case 1158:
            return read_csr(1158);
        case 1159:
            return read_csr(1159);
        case 1160:
            return read_csr(1160);
        case 1161:
            return read_csr(1161);
        case 1162:
            return read_csr(1162);
        case 1163:
            return read_csr(1163);
        case 1164:
            return read_csr(1164);
        case 1165:
            return read_csr(1165);
        case 1166:
            return read_csr(1166);
        case 1167:
            return read_csr(1167);
        case 1168:
            return read_csr(1168);
        case 1169:
            return read_csr(1169);
        case 1170:
            return read_csr(1170);
        case 1171:
            return read_csr(1171);
        case 1172:
            return read_csr(1172);
        case 1173:
            return read_csr(1173);
        case 1174:
            return read_csr(1174);
        case 1175:
            return read_csr(1175);
        case 1176:
            return read_csr(1176);
        case 1177:
            return read_csr(1177);
        case 1178:
            return read_csr(1178);
        case 1179:
            return read_csr(1179);
        case 1180:
            return read_csr(1180);
        case 1181:
            return read_csr(1181);
        case 1182:
            return read_csr(1182);
        case 1183:
            return read_csr(1183);
        case 1184:
            return read_csr(1184);
        case 1185:
            return read_csr(1185);
        case 1186:
            return read_csr(1186);
        case 1187:
            return read_csr(1187);
        case 1188:
            return read_csr(1188);
        case 1189:
            return read_csr(1189);
        case 1190:
            return read_csr(1190);
        case 1191:
            return read_csr(1191);
        case 1192:
            return read_csr(1192);
        case 1193:
            return read_csr(1193);
        case 1194:
            return read_csr(1194);
        case 1195:
            return read_csr(1195);
        case 1196:
            return read_csr(1196);
        case 1197:
            return read_csr(1197);
        case 1198:
            return read_csr(1198);
        case 1199:
            return read_csr(1199);
        case 1200:
            return read_csr(1200);
        case 1201:
            return read_csr(1201);
        case 1202:
            return read_csr(1202);
        case 1203:
            return read_csr(1203);
#endif
    }
    return 0;
}

static void csrw_ss(uint32_t csr_address, uint32_t value) {
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
#ifdef CSR_LONG_ADDR_MODE
        case 992:
            write_csr(992, value);
            break;
        case 993:
            write_csr(993, value);
            break;
        case 994:
            write_csr(994, value);
            break;
        case 995:
            write_csr(995, value);
            break;
        case 996:
            write_csr(996, value);
            break;
        case 997:
            write_csr(997, value);
            break;
        case 998:
            write_csr(998, value);
            break;
        case 999:
            write_csr(999, value);
            break;
        case 1000:
            write_csr(1000, value);
            break;
        case 1001:
            write_csr(1001, value);
            break;
        case 1002:
            write_csr(1002, value);
            break;
        case 1003:
            write_csr(1003, value);
            break;
        case 1004:
            write_csr(1004, value);
            break;
        case 1005:
            write_csr(1005, value);
            break;
        case 1006:
            write_csr(1006, value);
            break;
        case 1007:
            write_csr(1007, value);
            break;
        case 1008:
            write_csr(1008, value);
            break;
        case 1009:
            write_csr(1009, value);
            break;
        case 1010:
            write_csr(1010, value);
            break;
        case 1011:
            write_csr(1011, value);
            break;
        case 1012:
            write_csr(1012, value);
            break;
        case 1013:
            write_csr(1013, value);
            break;
        case 1014:
            write_csr(1014, value);
            break;
        case 1015:
            write_csr(1015, value);
            break;
        case 1016:
            write_csr(1016, value);
            break;
        case 1017:
            write_csr(1017, value);
            break;
        case 1018:
            write_csr(1018, value);
            break;
        case 1019:
            write_csr(1019, value);
            break;
        case 1020:
            write_csr(1020, value);
            break;
        case 1021:
            write_csr(1021, value);
            break;
        case 1022:
            write_csr(1022, value);
            break;
        case 1023:
            write_csr(1023, value);
            break;
        case 1024:
            write_csr(1024, value);
            break;
        case 1025:
            write_csr(1025, value);
            break;
        case 1026:
            write_csr(1026, value);
            break;
        case 1027:
            write_csr(1027, value);
            break;
        case 1028:
            write_csr(1028, value);
            break;
        case 1029:
            write_csr(1029, value);
            break;
        case 1030:
            write_csr(1030, value);
            break;
        case 1031:
            write_csr(1031, value);
            break;
        case 1032:
            write_csr(1032, value);
            break;
        case 1033:
            write_csr(1033, value);
            break;
        case 1034:
            write_csr(1034, value);
            break;
        case 1035:
            write_csr(1035, value);
            break;
        case 1036:
            write_csr(1036, value);
            break;
        case 1037:
            write_csr(1037, value);
            break;
        case 1038:
            write_csr(1038, value);
            break;
        case 1039:
            write_csr(1039, value);
            break;
        case 1040:
            write_csr(1040, value);
            break;
        case 1041:
            write_csr(1041, value);
            break;
        case 1042:
            write_csr(1042, value);
            break;
        case 1043:
            write_csr(1043, value);
            break;
        case 1044:
            write_csr(1044, value);
            break;
        case 1045:
            write_csr(1045, value);
            break;
        case 1046:
            write_csr(1046, value);
            break;
        case 1047:
            write_csr(1047, value);
            break;
        case 1048:
            write_csr(1048, value);
            break;
        case 1049:
            write_csr(1049, value);
            break;
        case 1050:
            write_csr(1050, value);
            break;
        case 1051:
            write_csr(1051, value);
            break;
        case 1052:
            write_csr(1052, value);
            break;
        case 1053:
            write_csr(1053, value);
            break;
        case 1054:
            write_csr(1054, value);
            break;
        case 1055:
            write_csr(1055, value);
            break;
        case 1056:
            write_csr(1056, value);
            break;
        case 1057:
            write_csr(1057, value);
            break;
        case 1058:
            write_csr(1058, value);
            break;
        case 1059:
            write_csr(1059, value);
            break;
        case 1060:
            write_csr(1060, value);
            break;
        case 1061:
            write_csr(1061, value);
            break;
        case 1062:
            write_csr(1062, value);
            break;
        case 1063:
            write_csr(1063, value);
            break;
        case 1064:
            write_csr(1064, value);
            break;
        case 1065:
            write_csr(1065, value);
            break;
        case 1066:
            write_csr(1066, value);
            break;
        case 1067:
            write_csr(1067, value);
            break;
        case 1068:
            write_csr(1068, value);
            break;
        case 1069:
            write_csr(1069, value);
            break;
        case 1070:
            write_csr(1070, value);
            break;
        case 1071:
            write_csr(1071, value);
            break;
        case 1072:
            write_csr(1072, value);
            break;
        case 1073:
            write_csr(1073, value);
            break;
        case 1074:
            write_csr(1074, value);
            break;
        case 1075:
            write_csr(1075, value);
            break;
        case 1076:
            write_csr(1076, value);
            break;
        case 1077:
            write_csr(1077, value);
            break;
        case 1078:
            write_csr(1078, value);
            break;
        case 1079:
            write_csr(1079, value);
            break;
        case 1080:
            write_csr(1080, value);
            break;
        case 1081:
            write_csr(1081, value);
            break;
        case 1082:
            write_csr(1082, value);
            break;
        case 1083:
            write_csr(1083, value);
            break;
        case 1084:
            write_csr(1084, value);
            break;
        case 1085:
            write_csr(1085, value);
            break;
        case 1086:
            write_csr(1086, value);
            break;
        case 1087:
            write_csr(1087, value);
            break;
        case 1088:
            write_csr(1088, value);
            break;
        case 1089:
            write_csr(1089, value);
            break;
        case 1090:
            write_csr(1090, value);
            break;
        case 1091:
            write_csr(1091, value);
            break;
        case 1092:
            write_csr(1092, value);
            break;
        case 1093:
            write_csr(1093, value);
            break;
        case 1094:
            write_csr(1094, value);
            break;
        case 1095:
            write_csr(1095, value);
            break;
        case 1096:
            write_csr(1096, value);
            break;
        case 1097:
            write_csr(1097, value);
            break;
        case 1098:
            write_csr(1098, value);
            break;
        case 1099:
            write_csr(1099, value);
            break;
        case 1100:
            write_csr(1100, value);
            break;
        case 1101:
            write_csr(1101, value);
            break;
        case 1102:
            write_csr(1102, value);
            break;
        case 1103:
            write_csr(1103, value);
            break;
        case 1104:
            write_csr(1104, value);
            break;
        case 1105:
            write_csr(1105, value);
            break;
        case 1106:
            write_csr(1106, value);
            break;
        case 1107:
            write_csr(1107, value);
            break;
        case 1108:
            write_csr(1108, value);
            break;
        case 1109:
            write_csr(1109, value);
            break;
        case 1110:
            write_csr(1110, value);
            break;
        case 1111:
            write_csr(1111, value);
            break;
        case 1112:
            write_csr(1112, value);
            break;
        case 1113:
            write_csr(1113, value);
            break;
        case 1114:
            write_csr(1114, value);
            break;
        case 1115:
            write_csr(1115, value);
            break;
        case 1116:
            write_csr(1116, value);
            break;
        case 1117:
            write_csr(1117, value);
            break;
        case 1118:
            write_csr(1118, value);
            break;
        case 1119:
            write_csr(1119, value);
            break;
        case 1120:
            write_csr(1120, value);
            break;
        case 1121:
            write_csr(1121, value);
            break;
        case 1122:
            write_csr(1122, value);
            break;
        case 1123:
            write_csr(1123, value);
            break;
        case 1124:
            write_csr(1124, value);
            break;
        case 1125:
            write_csr(1125, value);
            break;
        case 1126:
            write_csr(1126, value);
            break;
        case 1127:
            write_csr(1127, value);
            break;
        case 1128:
            write_csr(1128, value);
            break;
        case 1129:
            write_csr(1129, value);
            break;
        case 1130:
            write_csr(1130, value);
            break;
        case 1131:
            write_csr(1131, value);
            break;
        case 1132:
            write_csr(1132, value);
            break;
        case 1133:
            write_csr(1133, value);
            break;
        case 1134:
            write_csr(1134, value);
            break;
        case 1135:
            write_csr(1135, value);
            break;
        case 1136:
            write_csr(1136, value);
            break;
        case 1137:
            write_csr(1137, value);
            break;
        case 1138:
            write_csr(1138, value);
            break;
        case 1139:
            write_csr(1139, value);
            break;
        case 1140:
            write_csr(1140, value);
            break;
        case 1141:
            write_csr(1141, value);
            break;
        case 1142:
            write_csr(1142, value);
            break;
        case 1143:
            write_csr(1143, value);
            break;
        case 1144:
            write_csr(1144, value);
            break;
        case 1145:
            write_csr(1145, value);
            break;
        case 1146:
            write_csr(1146, value);
            break;
        case 1147:
            write_csr(1147, value);
            break;
        case 1148:
            write_csr(1148, value);
            break;
        case 1149:
            write_csr(1149, value);
            break;
        case 1150:
            write_csr(1150, value);
            break;
        case 1151:
            write_csr(1151, value);
            break;
        case 1152:
            write_csr(1152, value);
            break;
        case 1153:
            write_csr(1153, value);
            break;
        case 1154:
            write_csr(1154, value);
            break;
        case 1155:
            write_csr(1155, value);
            break;
        case 1156:
            write_csr(1156, value);
            break;
        case 1157:
            write_csr(1157, value);
            break;
        case 1158:
            write_csr(1158, value);
            break;
        case 1159:
            write_csr(1159, value);
            break;
        case 1160:
            write_csr(1160, value);
            break;
        case 1161:
            write_csr(1161, value);
            break;
        case 1162:
            write_csr(1162, value);
            break;
        case 1163:
            write_csr(1163, value);
            break;
        case 1164:
            write_csr(1164, value);
            break;
        case 1165:
            write_csr(1165, value);
            break;
        case 1166:
            write_csr(1166, value);
            break;
        case 1167:
            write_csr(1167, value);
            break;
        case 1168:
            write_csr(1168, value);
            break;
        case 1169:
            write_csr(1169, value);
            break;
        case 1170:
            write_csr(1170, value);
            break;
        case 1171:
            write_csr(1171, value);
            break;
        case 1172:
            write_csr(1172, value);
            break;
        case 1173:
            write_csr(1173, value);
            break;
        case 1174:
            write_csr(1174, value);
            break;
        case 1175:
            write_csr(1175, value);
            break;
        case 1176:
            write_csr(1176, value);
            break;
        case 1177:
            write_csr(1177, value);
            break;
        case 1178:
            write_csr(1178, value);
            break;
        case 1179:
            write_csr(1179, value);
            break;
        case 1180:
            write_csr(1180, value);
            break;
        case 1181:
            write_csr(1181, value);
            break;
        case 1182:
            write_csr(1182, value);
            break;
        case 1183:
            write_csr(1183, value);
            break;
        case 1184:
            write_csr(1184, value);
            break;
        case 1185:
            write_csr(1185, value);
            break;
        case 1186:
            write_csr(1186, value);
            break;
        case 1187:
            write_csr(1187, value);
            break;
        case 1188:
            write_csr(1188, value);
            break;
        case 1189:
            write_csr(1189, value);
            break;
        case 1190:
            write_csr(1190, value);
            break;
        case 1191:
            write_csr(1191, value);
            break;
        case 1192:
            write_csr(1192, value);
            break;
        case 1193:
            write_csr(1193, value);
            break;
        case 1194:
            write_csr(1194, value);
            break;
        case 1195:
            write_csr(1195, value);
            break;
        case 1196:
            write_csr(1196, value);
            break;
        case 1197:
            write_csr(1197, value);
            break;
        case 1198:
            write_csr(1198, value);
            break;
        case 1199:
            write_csr(1199, value);
            break;
        case 1200:
            write_csr(1200, value);
            break;
        case 1201:
            write_csr(1201, value);
            break;
        case 1202:
            write_csr(1202, value);
            break;
        case 1203:
            write_csr(1203, value);
            break;
#endif
    }
}

#endif  // CSR_H
