#pragma once

#include <juce_dsp/juce_dsp.h>

#include "PitchDetector.h"
#include "PsolaShifter.h"

#include <atomic>
#include <cmath>

namespace fx
{
// Autotune, pitch/formant shift and a two-voice harmonizer sharing one pitch detector.
// A voice is mono, so the block is mixed down, processed once and copied back to
// every channel.
class PitchEngine
{
public:
    struct Settings
    {
        bool  tuneOn = false;
        int   key = 0;            // 0 = C ... 11 = B
        int   scale = 0;          // index into getScaleNames()
        float retuneMs = 20.0f;   // 0 = instant ("hard tune")
        float amount = 1.0f;      // 0..1 of the correction applied

        bool  shiftOn = false;
        float semitones = 0.0f;
        float formantSemitones = 0.0f;

        bool  harmonyOn = false;
        float interval1 = 4.0f, interval2 = 7.0f; // semitones relative to the lead
        float harmonyMix = 0.5f;

        bool isActive() const noexcept { return tuneOn || shiftOn || harmonyOn; }
    };

    static const juce::StringArray& getKeyNames()
    {
        static const juce::StringArray names { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
        return names;
    }

    static const juce::StringArray& getScaleNames()
    {
        static const juce::StringArray names { "Chromatic", "Major", "Minor", "Harmonic Minor",
                                               "Major Pentatonic", "Minor Pentatonic", "Blues" };
        return names;
    }

    void prepare (const juce::dsp::ProcessSpec& spec)
    {
        detector.prepare (spec.sampleRate);
        lead.prepare (spec.sampleRate);
        for (auto& voice : harmonies)
            voice.prepare (spec.sampleRate);

        mono.assign (spec.maximumBlockSize, 0.0f);
        sampleRate = spec.sampleRate;
        reset();
    }

    void reset()
    {
        detector.reset();
        lead.reset();
        for (auto& voice : harmonies)
            voice.reset();

        correction = 0.0f;
        targetNote = -1;
        leadWasOn = harmonyWasOn = false;
        detectedNote.store (-1.0f);
        tunedNote.store (-1);
    }

    void process (const juce::dsp::ProcessContextReplacing<float>& context, const Settings& settings) noexcept
    {
        auto& block = context.getOutputBlock();
        const auto numSamples  = (int) block.getNumSamples();
        const auto numChannels = block.getNumChannels();
        auto* voice = mono.data();

        jassert ((size_t) numSamples <= mono.size());

        juce::FloatVectorOperations::copy (voice, block.getChannelPointer (0), numSamples);
        for (size_t ch = 1; ch < numChannels; ++ch)
            juce::FloatVectorOperations::add (voice, block.getChannelPointer (ch), numSamples);
        if (numChannels > 1)
            juce::FloatVectorOperations::multiply (voice, 1.0f / (float) numChannels, numSamples);

        detector.push (voice, numSamples);
        const auto frequency = detector.getFrequency();
        const auto period = detector.getPeriodInSamples();

        updateCorrection (settings, frequency, (float) numSamples / (float) sampleRate);

        const auto shift = settings.shiftOn ? settings.semitones : 0.0f;
        const auto formantRatio = settings.shiftOn ? std::exp2 (settings.formantSemitones / 12.0f) : 1.0f;
        const auto leadRatio = std::exp2 ((correction + shift) / 12.0f);
        const auto leadOn = settings.tuneOn || settings.shiftOn;

        // Clear stale audio when a voice is switched back on.
        if (leadOn && ! leadWasOn)
            lead.reset();
        if (settings.harmonyOn && ! harmonyWasOn)
            for (auto& harmony : harmonies)
                harmony.reset();

        leadWasOn = leadOn;
        harmonyWasOn = settings.harmonyOn;

        lead.setParameters (leadRatio, formantRatio, period);
        harmonies[0].setParameters (leadRatio * std::exp2 (settings.interval1 / 12.0f), formantRatio, period);
        harmonies[1].setParameters (leadRatio * std::exp2 (settings.interval2 / 12.0f), formantRatio, period);

        const auto harmonyGain = 0.7f * settings.harmonyMix;

        for (int i = 0; i < numSamples; ++i)
        {
            const auto input = voice[i];
            auto output = leadOn ? lead.processSample (input) : input;

            if (settings.harmonyOn)
                output += harmonyGain * (harmonies[0].processSample (input) + harmonies[1].processSample (input));

            voice[i] = output;
        }

        for (size_t ch = 0; ch < numChannels; ++ch)
            juce::FloatVectorOperations::copy (block.getChannelPointer (ch), voice, numSamples);
    }

    // For the UI tuner: the sung note (fractional MIDI, -1 when unvoiced) and the
    // note autotune is pulling towards (-1 when inactive).
    std::atomic<float> detectedNote { -1.0f };
    std::atomic<int> tunedNote { -1 };

private:
    static constexpr int scaleMask (std::initializer_list<int> degrees)
    {
        int mask = 0;
        for (auto degree : degrees)
            mask |= 1 << degree;
        return mask;
    }

    static int getScaleMask (int scale) noexcept
    {
        static constexpr int masks[] = {
            0xfff,                                    // Chromatic
            scaleMask ({ 0, 2, 4, 5, 7, 9, 11 }),     // Major
            scaleMask ({ 0, 2, 3, 5, 7, 8, 10 }),     // Minor
            scaleMask ({ 0, 2, 3, 5, 7, 8, 11 }),     // Harmonic minor
            scaleMask ({ 0, 2, 4, 7, 9 }),            // Major pentatonic
            scaleMask ({ 0, 3, 5, 7, 10 }),           // Minor pentatonic
            scaleMask ({ 0, 3, 5, 6, 7, 10 }),        // Blues
        };

        return masks[juce::jlimit (0, (int) std::size (masks) - 1, scale)];
    }

    int nearestNoteInScale (float note, int key, int scale) const noexcept
    {
        const auto mask = getScaleMask (scale);
        const auto inScale = [&] (int n) { return ((mask >> (((n - key) % 12 + 12) % 12)) & 1) != 0; };

        auto best = -1;
        auto bestDistance = 100.0f;
        const auto base = (int) std::floor (note);

        for (auto n = base - 6; n <= base + 7; ++n)
        {
            const auto distance = std::abs (note - (float) n);

            if (inScale (n) && distance < bestDistance)
            {
                best = n;
                bestDistance = distance;
            }
        }

        // Hysteresis: stay on the current note until another is clearly closer,
        // so notes sung between two pitches don't warble back and forth.
        constexpr auto hysteresis = 0.2f;

        if (targetNote >= 0 && inScale (targetNote) && std::abs (note - (float) targetNote) < bestDistance + hysteresis)
            return targetNote;

        return best;
    }

    void updateCorrection (const Settings& settings, float frequency, float blockSeconds) noexcept
    {
        auto desired = 0.0f;

        if (settings.tuneOn && frequency > 0.0f)
        {
            const auto note = 69.0f + 12.0f * std::log2 (frequency / 440.0f);
            targetNote = nearestNoteInScale (note, settings.key, settings.scale);
            desired = ((float) targetNote - note) * settings.amount;
            detectedNote.store (note);
            tunedNote.store (targetNote);
        }
        else
        {
            targetNote = -1;
            detectedNote.store (frequency > 0.0f ? 69.0f + 12.0f * std::log2 (frequency / 440.0f) : -1.0f);
            tunedNote.store (-1);
        }

        const auto coefficient = settings.retuneMs < 0.5f ? 1.0f
                                                          : 1.0f - std::exp (-blockSeconds * 1000.0f / settings.retuneMs);
        correction += (desired - correction) * coefficient;
    }

    PitchDetector detector;
    PsolaShifter lead;
    std::array<PsolaShifter, 2> harmonies;

    std::vector<float> mono;
    double sampleRate = 44100.0;
    float correction = 0.0f;
    int targetNote = -1;
    bool leadWasOn = false, harmonyWasOn = false;
};
} // namespace fx
