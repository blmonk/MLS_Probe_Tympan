/*
 * MLS_Probe_Tympan
 *
 * Created: Bryan Monk, June 2026
 * Purpose: Emit a Maximum Length Sequence (MLS) from the left earpiece receiver and
 *          record both the MLS reference signal and the left-earpiece PDM microphone
 *          response to a 2-channel WAV file on the SD card. MLS length is selectable
 *          at runtime from 1023 up to 65535 samples (2^n - 1); default is 1023.
 *
 *          Channel 0 (WAV left):  averaged left-earpiece PDM mic signal
 *          Channel 1 (WAV right): MLS reference signal sent to the speaker
 *
 * Hardware: Tympan RevE + EarpieceShield
 * Sample rate: 48000 Hz, block size: 128 (required for SD recording)
 *
 * Serial Monitor:
 *   Send 'h' for the help menu.
 *   Key commands: p = start MLS, P = stop MLS, r = start recording, s = stop recording
 *   Amplitude: 'a <val>' sets A in [0.0, 1.0]; '+'/'-' nudge by 0.05
 *   Length: 'len <val>' sets the MLS length, snapping to the nearest valid value
 *
 * MIT License. Use at your own risk.
 */

#include <Tympan_Library.h>
#include "AudioSynthMLS_F32.h"

// -------- Audio settings --------
// NOTE: audio_block_samples MUST be 128 for AudioSDWriter_F32.
const float sample_rate_Hz      = 48000.0f;
const int   audio_block_samples = 128;
AudioSettings_F32 audio_settings(sample_rate_Hz, audio_block_samples);

// -------- Hardware --------
Tympan         myTympan(TympanRev::E, audio_settings);
EarpieceShield earpieceShield(TympanRev::E, AICShieldRev::A);

// -------- Audio objects --------
AudioSynthMLS_F32       mlsSource(audio_settings);      // MLS signal generator
AudioInputI2SQuad_F32   i2s_in(audio_settings);         // PDM mic inputs (4 channels)
AudioMixer4_F32         mic_mixer(audio_settings);       // average left front + rear mics
AudioOutputI2SQuad_F32  i2s_out(audio_settings);        // speaker outputs
AudioSDWriter_F32       audioSDWriter(audio_settings);   // 2-channel WAV recorder

// -------- Audio connections --------
//   MLS -> left earpiece receiver and Tympan headphone jack (for monitoring)
AudioConnection_F32  pc1(mlsSource,  0, i2s_out, EarpieceShield::OUTPUT_LEFT_EARPIECE);
AudioConnection_F32  pc2(mlsSource,  0, i2s_out, EarpieceShield::OUTPUT_LEFT_TYMPAN);

//   Left earpiece PDM mics -> mixer (front to ch0, rear to ch1)
AudioConnection_F32  pc3(i2s_in, EarpieceShield::PDM_LEFT_FRONT, mic_mixer, 0);
AudioConnection_F32  pc4(i2s_in, EarpieceShield::PDM_LEFT_REAR,  mic_mixer, 1);

//   SD writer: ch0 = mic (left earpiece averaged), ch1 = MLS reference
AudioConnection_F32  pc5(mic_mixer, 0, audioSDWriter, 0);
AudioConnection_F32  pc6(mlsSource, 0, audioSDWriter, 1);

// -------- Serial manager --------
#include "SerialManager.h"
SerialManager serialManager;

// -------- Globals --------
float input_gain_dB           = 20.0f;
float output_volume_dB        = 0.0f;
bool  enable_printCPUandMemory = false;


// ================================================================
void setup() {
    myTympan.beginBothSerial(); delay(1000);
    myTympan.println("MLS_Probe_Tympan: Starting setup()...");
    myTympan.print("  Sample rate (Hz):  "); myTympan.println(sample_rate_Hz);
    myTympan.print("  Block size:        "); myTympan.println(audio_block_samples);
    myTympan.print("  MLS length:        "); myTympan.println(mlsSource.getLength());

    AudioMemory_F32(40, audio_settings);

    // Enable hardware
    myTympan.enable();
    earpieceShield.enable();

    // Use PDM mics in the earpieces
    myTympan.enableDigitalMicInputs(true);
    earpieceShield.enableDigitalMicInputs(true);

    // Average left front and rear mics; silence unused mixer inputs
    mic_mixer.gain(0, 0.5f);
    mic_mixer.gain(1, 0.5f);
    mic_mixer.gain(2, 0.0f);
    mic_mixer.gain(3, 0.0f);

    // Set initial audio levels
    myTympan.volume_dB(output_volume_dB);
    earpieceShield.volume_dB(output_volume_dB);
    myTympan.setInputGain_dB(input_gain_dB);

    // Configure SD writer (2-ch WAV: ch0 = mic, ch1 = MLS reference)
    audioSDWriter.setSerial(&myTympan);
    audioSDWriter.setNumWriteChannels(2);
    myTympan.println("  SD: 2-channel WAV (ch0 = mic, ch1 = MLS reference)");

    // MLS starts disabled; default amplitude 0.5
    mlsSource.setAmplitude(0.5f);
    mlsSource.setEnable(false);

    myTympan.println("Setup complete.");
    serialManager.printHelp();
}


// ================================================================
void loop() {
    // Process serial commands
    while (Serial.available()) serialManager.respondToByte((char)Serial.read());

    // Service SD write buffer — must be called frequently from loop()
    audioSDWriter.serviceSD_withWarnings(i2s_in);

    // Blink LED fast while recording, slow otherwise
    myTympan.serviceLEDs(millis(),
        audioSDWriter.getState() == AudioSDWriter::STATE::RECORDING);

    // Optional CPU/memory diagnostics
    if (enable_printCPUandMemory)
        myTympan.printCPUandMemory(millis(), 3000);

    // Volume potentiometer
    servicePotentiometer(millis(), 100);
}


// ================================================================
// Helper functions (also called from SerialManager via extern declarations)

void servicePotentiometer(unsigned long curTime_millis,
                          unsigned long updatePeriod_millis) {
    static unsigned long lastUpdate_millis = 0;
    static float prev_val = -1.0f;

    if (curTime_millis < lastUpdate_millis) lastUpdate_millis = 0;
    if ((curTime_millis - lastUpdate_millis) <= updatePeriod_millis) return;
    lastUpdate_millis = curTime_millis;

    float val = float(myTympan.readPotentiometer()) / 1023.0f;
    val = (1.0f / 8.0f) * float(int(8.0f * val + 0.5f));  // quantize to 9 steps

    if (abs(val - prev_val) > 0.05f) {
        prev_val = val;
        const float min_vol = -20.0f, max_vol = 20.0f;
        setOutputVolume_dB(min_vol + (max_vol - min_vol) * val);
    }
}

void setOutputVolume_dB(float vol_dB) {
    output_volume_dB = max(-63.6f, min(vol_dB, 24.0f));
    myTympan.volume_dB(output_volume_dB);
    earpieceShield.volume_dB(output_volume_dB);
    Serial.print("Output volume (dB): "); Serial.println(output_volume_dB, 1);
}

float setInputGain_dB(float gain_dB) {
    input_gain_dB = max(0.0f, min(gain_dB, 47.5f));
    myTympan.setInputGain_dB(input_gain_dB);
    Serial.print("Input gain (dB): "); Serial.println(input_gain_dB, 1);
    return input_gain_dB;
}

void printSettings(void) {
    Serial.println("--- Current Settings ---");
    Serial.print("  MLS playback:   "); Serial.println(mlsSource.getEnable() ? "ON" : "OFF");
    Serial.print("  MLS length:     "); Serial.print(mlsSource.getLength());
    Serial.print(" (degree ");         Serial.print(mlsSource.getDegree());
    Serial.print(", period ");         Serial.print(1000.0f * mlsSource.getLength() / sample_rate_Hz, 1);
    Serial.println(" ms)");
    Serial.print("  Amplitude A:    "); Serial.println(mlsSource.getAmplitude(), 4);
    Serial.print("  Input gain:     "); Serial.print(input_gain_dB, 1);    Serial.println(" dB");
    Serial.print("  Output volume:  "); Serial.print(output_volume_dB, 1); Serial.println(" dB");
    Serial.print("  SD state:       ");
    if (audioSDWriter.getState() == AudioSDWriter::STATE::RECORDING) {
        Serial.println("RECORDING -> " + audioSDWriter.getCurrentFilename());
    } else {
        Serial.println("stopped");
    }
}
