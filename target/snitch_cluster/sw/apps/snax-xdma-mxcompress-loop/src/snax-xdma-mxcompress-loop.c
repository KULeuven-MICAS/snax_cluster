// Copyright 2026 KU Leuven. SPDX-License-Identifier: Apache-2.0
//
// In-transit MX compression loopback: BF16 --StreamCastRt(compress, reader ext)--> MXFP4 across the
// reader->writer path (the mimicked "link") --StreamDecompressRt(decompress, writer ext)--> BF16.
// Models compress-at-sender / decompress-at-receiver on a single cluster. Measures the transfer cycles
// vs a plain BF16 copy and checks exact recovery. The link-BW win is the beat ratio (baseline vs
// compressed link beats); the local loopback writes the recovered BF16 to L1, so its *cycles* are
// L1-bound (equal cycles => the in-flight compress+decompress is FREE under the transfer).
#include "data.h"
#include "snax-xdma-lib.h"
#include "snrt.h"

int main() {
    int err = 0;
    uint32_t tcdm = snrt_cluster_base_addrl();
    uint16_t *in  = (uint16_t *)tcdm;
    uint16_t *out = (uint16_t *)(tcdm + N * 2 + 512); // BF16 output, offset past the input

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(in, bf16_input, N * 2);
        snrt_dma_wait_all();

        // ---- baseline: plain BF16 copy (all extensions off) ----
        for (int e = 0; e < XDMA_SRC_EXT_NUM; e++) snax_xdma_disable_src_ext(e);
        for (int e = 0; e < XDMA_DST_EXT_NUM; e++) snax_xdma_disable_dst_ext(e);
        snax_xdma_memcpy_1d((void *)in, (void *)out, N * 2);
        int t0 = snax_xdma_start();
        snax_xdma_local_wait(t0);
        uint32_t base_cyc = snax_xdma_last_task_cycle();

        // ---- loopback: compress at the reader (StreamCastRt) + decompress at the writer (StreamDecompressRt) ----
        // csr0 = srcFmt(BF16=1) | dstFmt(MXFP4=7)<<2 | emitScale/hasScale(1)<<5
        uint32_t cast_csr[1] = {1u | (7u << 2) | (1u << 5)};
        uint32_t deco_csr[1] = {1u | (7u << 2) | (1u << 5)};
        for (int e = 0; e < XDMA_SRC_EXT_NUM; e++)
            if (e != READER_EXT_STREAMCASTRT) snax_xdma_disable_src_ext(e);
        snax_xdma_enable_src_ext(READER_EXT_STREAMCASTRT, cast_csr);
        for (int e = 0; e < XDMA_DST_EXT_NUM; e++)
            if (e != WRITER_EXT_STREAMDECOMPRESSRT) snax_xdma_disable_dst_ext(e);
        snax_xdma_enable_dst_ext(WRITER_EXT_STREAMDECOMPRESSRT, deco_csr);
        snax_xdma_memcpy_1d((void *)in, (void *)out, N * 2);
        int t1 = snax_xdma_start();
        snax_xdma_local_wait(t1);
        uint32_t loop_cyc = snax_xdma_last_task_cycle();

        // ---- report ----
        uint32_t base_beats = (N * 2 + 63) / 64;   // 64B beats to move N BF16
        uint32_t link_beats = (N / 2048) * 17;     // MXFP4 + scaleBurst=16: 16 data + 1 scale per 2048 BF16
        printf("[mxcompress-loop] N=%d BF16.  baseline copy = %u cyc;  compress->decompress loopback = %u cyc\n",
               N, base_cyc, loop_cyc);
        printf("[mxcompress-loop] LINK beats: baseline %u -> compressed %u  (%u.%02ux fewer on the link)\n",
               base_beats, link_beats, base_beats / link_beats, (base_beats * 100u / link_beats) % 100u);

        // ---- recovery diagnostic: categorize mismatches (zero-out vs nonzero-wrong) + per-block histogram ----
        int nzero = 0, first_bad_block = -1, mis_per_block[64];
        for (int b = 0; b < 64; b++) mis_per_block[b] = 0;
        for (int i = 0; i < N; i++)
            if (out[i] != bf16_input[i]) {
                err++;
                mis_per_block[i >> 5]++;
                if (out[i] == 0) nzero++;
                if (first_bad_block < 0) first_bad_block = i >> 5;
            }
        printf("[mxcompress-loop] recovery: %s  mism=%d  zero-out=%d  nonzero-wrong=%d\n",
               err ? "FAIL" : "EXACT", err, nzero, err - nzero);
        printf("[mxcompress-loop] mism per block (of 64):");
        for (int b = 0; b < 64; b++)
            if (mis_per_block[b]) printf(" b%d=%d", b, mis_per_block[b]);
        printf("\n");
        if (first_bad_block >= 0) { // full in->out dump of the first failing block
            int b = first_bad_block;
            printf("[mxcompress-loop] block %d full dump (elem: in -> out):\n", b);
            for (int j = 0; j < 32; j++) {
                int i = b * 32 + j;
                printf("  [%2d] %04x -> %04x%s\n", j, bf16_input[i], out[i],
                       out[i] != bf16_input[i] ? "  X" : "");
            }
        }
    }
    return err;
}
