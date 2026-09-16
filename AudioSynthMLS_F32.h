/*
 * AudioSynthMLS_F32
 *
 * Generates a 1023-sample Maximum Length Sequence (degree-10 Fibonacci LFSR,
 * primitive polynomial x^10 + x^7 + 1) as a continuously looping audio source.
 *
 * Binary LFSR outputs {0, 1} are mapped to {-1.0, +1.0} and scaled by amplitude A.
 * Sequence resets to index 0 each time playback is enabled from a stopped state,
 * providing a deterministic start for each measurement session.
 *
 * MIT License. Use at your own risk.
 */

#ifndef _AudioSynthMLS_F32_h
#define _AudioSynthMLS_F32_h

#include <Tympan_Library.h>

class AudioSynthMLS_F32 : public AudioStream_F32 {
    //GUI: inputs:0, outputs:1
public:
    static const int MLS_LENGTH = 1023;  // 2^10 - 1

    AudioSynthMLS_F32(const AudioSettings_F32 &settings)
        : AudioStream_F32(0, NULL)
    {
        generateSequence();
    }

    // Set amplitude A in [0.0, 1.0]. Returns the clamped value actually set.
    float setAmplitude(float a) {
        if (a < 0.0f) a = 0.0f;
        if (a > 1.0f) a = 1.0f;
        amplitude = a;
        return amplitude;
    }
    float getAmplitude(void) const { return amplitude; }

    // Enable/disable output. Enabling from a stopped state resets the sequence index.
    void setEnable(bool en) {
        if (en && !enabled) seq_index = 0;
        enabled = en;
    }
    bool getEnable(void) const { return enabled; }

    void update(void) override {
        audio_block_f32_t *block = AudioStream_F32::allocate_f32();
        if (!block) return;

        float *dp = block->data;
        int    n  = block->length;

        if (enabled) {
            float A = amplitude;  // local copy — avoids repeated volatile reads in loop
            for (int i = 0; i < n; i++) {
                dp[i] = A * mls[seq_index];
                if (++seq_index >= MLS_LENGTH) seq_index = 0;
            }
        } else {
            for (int i = 0; i < n; i++) dp[i] = 0.0f;
        }

        AudioStream_F32::transmit(block, 0);
        AudioStream_F32::release(block);
    }

private:
    volatile float amplitude  = 0.5f;
    volatile bool  enabled    = false;
    volatile int   seq_index  = 0;
    float          mls[MLS_LENGTH];

    /*
     * Fibonacci right-shift LFSR, degree 10, primitive polynomial x^10 + x^7 + 1.
     *
     * Register layout (0-indexed): r[9] ... r[1] r[0], where r[0] is the LSB.
     * Each step:
     *   output   = r[0]                 (LSB shifted out)
     *   feedback = r[7] XOR r[0]        (taps at bit positions 7 and 0, 0-indexed)
     *   new state: shift right by 1, insert feedback at r[9]
     *
     * Recurrence: a[n] = a[n-3] XOR a[n-10], characteristic polynomial x^10 + x^7 + 1.
     * Initial state 0x3FF (all ones) produces a period of exactly 1023 samples.
     */
    void generateSequence(void) {
        uint16_t lfsr = 0x3FF;
        for (int i = 0; i < MLS_LENGTH; i++) {
            uint16_t out      = lfsr & 1;
            uint16_t feedback = ((lfsr >> 7) ^ lfsr) & 1;  // r[7] XOR r[0]
            lfsr = (lfsr >> 1) | (uint16_t)(feedback << 9);
            mls[i] = (out == 0) ? -1.0f : 1.0f;
        }
    }
};

#endif  // _AudioSynthMLS_F32_h
