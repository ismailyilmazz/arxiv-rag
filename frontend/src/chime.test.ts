import { afterEach, describe, expect, it, vi } from "vitest";
import { armChime, playChime, resetChime } from "./chime";

class FakeContext {
  static made = 0;
  static last: FakeContext | null = null;
  currentTime = 0;
  destination = {};
  started: number[] = [];
  resume = vi.fn(() => Promise.resolve());
  constructor() {
    FakeContext.made += 1;
    FakeContext.last = this;
  }
  createGain() {
    return { gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: (node: unknown) => node };
  }
  createOscillator() {
    const self = this;
    return {
      frequency: { value: 0 },
      connect: (node: unknown) => node,
      start(this: { frequency: { value: number } }) {
        self.started.push(this.frequency.value);
      },
      stop: vi.fn(),
    };
  }
}

describe("chime", () => {
  afterEach(() => {
    resetChime();
    vi.unstubAllGlobals();
    FakeContext.made = 0;
    FakeContext.last = null;
  });

  it("stays silent and safe when the browser has no audio support", () => {
    vi.stubGlobal("AudioContext", undefined);
    expect(() => {
      armChime();
      playChime("done");
    }).not.toThrow();
  });

  it("never lets a broken audio engine stop the page", () => {
    class Broken {
      constructor() {
        throw new Error("NotSupportedError");
      }
    }
    vi.stubGlobal("AudioContext", Broken);
    expect(() => armChime()).not.toThrow();
    vi.stubGlobal("AudioContext", FakeContext);
    armChime();
    FakeContext.last!.createOscillator = () => {
      throw new Error("audio failed");
    };
    expect(() => playChime("done")).not.toThrow();
  });

  it("does not play before a click has unlocked audio", () => {
    vi.stubGlobal("AudioContext", FakeContext);
    playChime("done");
    expect(FakeContext.made).toBe(0);
  });

  it("plays a rising pair when a job finishes and a single low note when it fails", () => {
    vi.stubGlobal("AudioContext", FakeContext);
    armChime();
    armChime();
    playChime("done");
    playChime("failed");
    expect(FakeContext.made).toBe(1);
    expect(FakeContext.last?.resume).toHaveBeenCalled();
    expect(FakeContext.last?.started).toEqual([660, 880, 330]);
  });
});
