// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
#include "snrt.h"

int main(){
    // The purpose of this program is to test the cluster core can write to 
    // the uart using printf without the help of the host system
    if (snrt_is_dm_core()){
        printf("Hello from the cluster\t\n");
    }
    
}