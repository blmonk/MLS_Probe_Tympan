/*
 * AudioSynthMLS_F32
 *
 * Generates a Maximum Length Sequence (MLS) as a continuously looping audio
 * source. The length is selectable at runtime from a fixed set of verified
 * maximal-length Fibonacci LFSR configurations: degrees 10-16, i.e. lengths
 * 1023 through 65535 samples (2^n - 1). Requesting a length snaps to the
 * nearest supported value -- see kVariants below.
 *
 * Binary LFSR outputs {0, 1} are mapped to {-1.0, +1.0} and scaled by amplitude A.
 * Sequence resets to index 0 each time playback is enabled from a stopped state,
 * providing a deterministic start for each measurement session. Changing the
 * length likewise always stops playback first, so the buffer is never read
 * mid-regeneration.
 *
 * MIT License. Use at your own risk.
 */

#ifndef _AudioSynthMLS_F32_h
#define _AudioSynthMLS_F32_h

#include <Tympan_Library.h>

// Storage for the longest supported MLS (65535 samples * 4 bytes ~= 256KB).
// Placed in DMAMEM -- the Teensy 4.1's second RAM bank, otherwise mostly
// unused by this sketch -- rather than ordinary static RAM, matching the
// convention already used for other large buffers in Tympan_Library (e.g.
// the I2S DMA buffers in output_i2s_quad_F32.cpp). File scope (rather than
// a class member) because DMAMEM is a linker-section attribute, which GCC
// only honors reliably on variables with static storage duration declared
// this way, not on ordinary non-static class members. This buffer is shared
// (not a per-instance member), so it assumes a single AudioSynthMLS_F32
// instance -- true for this sketch (just "mlsSource"); a second instance
// would corrupt/overwrite the first's sequence.
static const int AUDIOSYNTHMLS_MAX_LENGTH = 65535;
DMAMEM static float audioSynthMLS_buffer[AUDIOSYNTHMLS_MAX_LENGTH];

class AudioSynthMLS_F32 : public AudioStream_F32 {
    //GUI: inputs:0, outputs:1
public:
    static const int MIN_MLS_LENGTH = 1023;                      // 2^10 - 1: default, and the floor
    static const int MAX_MLS_LENGTH = AUDIOSYNTHMLS_MAX_LENGTH;  // 2^16 - 1: ceiling

    // Supported (degree, length, tap_mask) combinations. Each was found by
    // brute-force simulation (not copied from a reference table): starting
    // from an all-ones seed, the LFSR below must visit all 2^n - 1 nonzero
    // states before returning to the seed. Degrees 12, 13, 14 and 16 need 4
    // XOR taps -- no 2-tap (trinomial) feedback exists for those degrees.
    // Degree 10 matches what this file has always used.
    //
    // LFSR structure: register r[0..n-1], r[0] = LSB = output.
    //   feedback  = XOR-parity(state & tap_mask)   -- tap_mask always includes bit 0
    //   new_state = (state >> 1) | (feedback << (n-1))
    struct MLSVariant { uint8_t degree; uint32_t length; uint32_t tap_mask; };
    static constexpr int NUM_VARIANTS = 7;
    static constexpr MLSVariant kVariants[NUM_VARIANTS] = {
        { 10,   1023, 0x0081 },
        { 11,   2047, 0x0005 },
        { 12,   4095, 0x0107 },
        { 13,   8191, 0x0027 },
        { 14,  16383, 0x1007 },
        { 15,  32767, 0x0003 },
        { 16,  65535, 0x100B },
    };

    AudioSynthMLS_F32(const AudioSettings_F32 &settings)
        : AudioStream_F32(0, NULL)
    {
        setLength(MIN_MLS_LENGTH);
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

    // Request an MLS length; snaps to the nearest entry in kVariants (ties
    // favor the larger) and regenerates the sequence for it. Always stops
    // playback first: update() only ever reads the buffer/length while
    // enabled, and enabled is held false for the whole regeneration below,
    // so a concurrent audio-block update() can never observe a half-written
    // buffer or a length that doesn't match what's actually in it -- no
    // locking needed. Returns the length actually applied.
    int setLength(int requestedLength) {
        const MLSVariant &v = nearestVariant(requestedLength);

        enabled = false;

        generateSequence(v.degree, v.tap_mask, v.length);
        mls_length = v.length;
        mls_degree = v.degree;
        seq_index  = 0;

        return mls_length;
    }
    int getLength(void) const { return mls_length; }
    int getDegree(void) const { return mls_degree; }

    void update(void) override {
        audio_block_f32_t *block = AudioStream_F32::allocate_f32();
        if (!block) return;

        float *dp = block->data;
        int    n  = block->length;

        if (enabled) {
            float A   = amplitude;    // local copies -- avoid repeated volatile reads in loop
            int   len = mls_length;
            for (int i = 0; i < n; i++) {
                dp[i] = A * audioSynthMLS_buffer[seq_index];
                if (++seq_index >= len) seq_index = 0;
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
    volatile int   mls_length = MIN_MLS_LENGTH;
    volatile int   mls_degree = 10;

    static long absDiff(int requested, uint32_t candidateLength) {
        long d = (long)requested - (long)candidateLength;
        return (d < 0) ? -d : d;
    }

    static const MLSVariant &nearestVariant(int requested) {
        const MLSVariant *best = &kVariants[0];
        long bestDist = absDiff(requested, best->length);
        for (int i = 1; i < NUM_VARIANTS; i++) {
            long dist = absDiff(requested, kVariants[i].length);
            if (dist < bestDist || (dist == bestDist && kVariants[i].length > best->length)) {
                best = &kVariants[i];
                bestDist = dist;
            }
        }
        return *best;
    }

    static int parity(uint32_t x) {
        int p = 0;
        while (x) { p ^= 1; x &= (x - 1); }
        return p;
    }

    // Fills audioSynthMLS_buffer[0..length-1] with a +-1.0 maximal-length
    // sequence for the given degree/tap mask. Only runs when the length
    // changes (not per audio sample), so a simple parity loop is fine here.
    static void generateSequence(int degree, uint32_t tap_mask, int length) {
        uint32_t lfsr = (uint32_t)length;  // all-ones seed: 2^degree - 1 == length
        for (int i = 0; i < length; i++) {
            uint32_t out      = lfsr & 1u;
            uint32_t feedback = (uint32_t)parity(lfsr & tap_mask);
            lfsr = (lfsr >> 1) | (feedback << (degree - 1));
            audioSynthMLS_buffer[i] = (out == 0) ? -1.0f : 1.0f;
        }
    }
};

#endif  // _AudioSynthMLS_F32_h
