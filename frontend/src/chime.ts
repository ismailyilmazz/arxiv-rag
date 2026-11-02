export type Tone = "done" | "failed";

const NOTES: Record<Tone, number[]> = { done: [660, 880], failed: [330] };

let context: AudioContext | null = null;

function audioClass(): typeof AudioContext | undefined {
  return window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

export function armChime() {
  try {
    const Audio = audioClass();
    if (!Audio) return;
    context = context ?? new Audio();
    context.resume().catch(() => undefined);
  } catch {
    context = null;
  }
}

export function playChime(tone: Tone) {
  const ctx = context;
  if (!ctx) return;
  try {
    play(ctx, tone);
  } catch {
    context = null;
  }
}

function play(ctx: AudioContext, tone: Tone) {
  NOTES[tone].forEach((frequency, i) => {
    const at = ctx.currentTime + i * 0.18;
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(0.2, at + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.35);
    oscillator.connect(gain).connect(ctx.destination);
    oscillator.start(at);
    oscillator.stop(at + 0.4);
  });
}

export function resetChime() {
  context = null;
}
