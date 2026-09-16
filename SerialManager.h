
#ifndef _SerialManager_h
#define _SerialManager_h

#include <Tympan_Library.h>
#include "AudioSynthMLS_F32.h"

// Provided by the main .ino
extern Tympan              myTympan;
extern AudioSynthMLS_F32   mlsSource;
extern AudioSDWriter_F32   audioSDWriter;
extern bool                enable_printCPUandMemory;
extern float               input_gain_dB;
extern void                setOutputVolume_dB(float);
extern float               setInputGain_dB(float);
extern void                printSettings(void);


class SerialManager : public SerialManagerBase {
public:
    void printHelp(void);
    bool processCharacter(char c);

    // Intercept newlines to process buffered line commands.
    virtual void respondToByte(char c) override {
        if (c == '\r' || c == '\n') {
            if (lineLen > 0) {
                lineBuf[lineLen] = '\0';
                lineLen = 0;
                processLine();
            }
            return;
        }
        SerialManagerBase::respondToByte(c);
    }

    float amplitudeStep = 0.05f;
    float gainStep_dB   = 3.0f;

private:
    char lineBuf[64];
    int  lineLen = 0;

    void processLine(void);
};


void SerialManager::printHelp(void) {
    Serial.println();
    Serial.println("=== MLS_Probe_Tympan Help ===");
    Serial.println("  h / ?         : Print this help menu");
    Serial.println("  g             : Print current settings");
    Serial.println("  c / C         : Enable / Disable CPU and memory printing");
    Serial.println();
    Serial.println("--- MLS Playback ---");
    Serial.println("  p             : Start MLS playback (sequence resets to index 0)");
    Serial.println("  P             : Stop MLS playback (outputs silence)");
    Serial.println("  a <val>       : Set amplitude A in [0.0, 1.0], e.g.  a 0.5");
    Serial.println("  + / -         : Increase / Decrease A by " + String(amplitudeStep, 2));
    Serial.println();
    Serial.println("--- SD Recording ---");
    Serial.println("  r             : Start SD recording (ch0 = mic, ch1 = MLS reference)");
    Serial.println("  s             : Stop SD recording");
    Serial.println();
    Serial.println("--- Audio Levels ---");
    Serial.println("  i / I         : Increase / Decrease input gain by " +
                   String(gainStep_dB, 0) + " dB");
    Serial.println("  v <val>       : Set output volume in dB, e.g.  v 5.0");
    Serial.println();
}


bool SerialManager::processCharacter(char c) {
    if (c == 8 || c == 127) {          // backspace
        if (lineLen > 0) lineLen--;
        return true;
    }
    if (lineLen < 63) lineBuf[lineLen++] = c;
    return true;
}


void SerialManager::processLine(void) {
    // ---- Multi-character commands (checked first) ----

    if (strncmp(lineBuf, "a ", 2) == 0) {
        float val = atof(lineBuf + 2);
        Serial.print("Set amplitude A = ");
        Serial.println(mlsSource.setAmplitude(val), 4);
        return;
    }
    if (strncmp(lineBuf, "v ", 2) == 0) {
        float val = atof(lineBuf + 2);
        setOutputVolume_dB(val);
        return;
    }

    // ---- Single-character commands ----

    if (lineBuf[1] != '\0') {
        Serial.print("Unknown command: "); Serial.println(lineBuf);
        return;
    }

    char c = lineBuf[0];
    switch (c) {

        // Help / info
        case 'h': case '?':
            printHelp();
            break;
        case 'g':
            printSettings();
            break;

        // CPU/memory
        case 'c':
            enable_printCPUandMemory = true;
            Serial.println("CPU/memory printing enabled.");
            break;
        case 'C':
            enable_printCPUandMemory = false;
            Serial.println("CPU/memory printing disabled.");
            break;

        // MLS playback
        case 'p':
            mlsSource.setEnable(true);
            Serial.println("MLS playback started. Amplitude A = " +
                           String(mlsSource.getAmplitude(), 4));
            break;
        case 'P':
            mlsSource.setEnable(false);
            Serial.println("MLS playback stopped.");
            break;
        case '+':
            Serial.print("Amplitude A = ");
            Serial.println(mlsSource.setAmplitude(mlsSource.getAmplitude() + amplitudeStep), 4);
            break;
        case '-':
            Serial.print("Amplitude A = ");
            Serial.println(mlsSource.setAmplitude(mlsSource.getAmplitude() - amplitudeStep), 4);
            break;

        // SD recording
        case 'r':
            audioSDWriter.startRecording();
            Serial.println("SD recording started: " + audioSDWriter.getCurrentFilename());
            break;
        case 's':
            audioSDWriter.stopRecording();
            Serial.println("SD recording stopped.");
            break;

        // Input gain
        case 'i':
            setInputGain_dB(input_gain_dB + gainStep_dB);
            break;
        case 'I':
            setInputGain_dB(input_gain_dB - gainStep_dB);
            break;

        default:
            Serial.print("Unknown command: "); Serial.println(c);
            break;
    }
}

#endif  // _SerialManager_h
