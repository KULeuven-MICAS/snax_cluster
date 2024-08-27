// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

<%def name="icache_cfg(prop)">\
  % for lw in cfg['hives']:
${lw['icache'][prop]}${',' if not loop.last else ''}\
  % endfor
</%def>\
${cfg['tcdm']['depth']}x${cfg['data_width']}m4s
${icache_cfg('depth')}x${icache_cfg('cacheline')}m4s
${icache_cfg('depth')}x${cfg['tag_width']}m4s
