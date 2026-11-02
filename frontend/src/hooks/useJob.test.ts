import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "../api/client";
import type { Job } from "../api/types";
import { text } from "../strings";
import { useJob } from "./useJob";

const done = { job_id: "j", kind: "document", status: "done", stage: "done", done: 1, total: 1, position: 0 } as Job;

afterEach(() => vi.restoreAllMocks());

describe("useJob", () => {
  it("keeps trying through a lost connection and recovers", async () => {
    const getJob = vi.spyOn(api, "getJob");
    getJob.mockRejectedValueOnce(new ApiError(0, "ağ")).mockRejectedValueOnce(new ApiError(503, "meşgul")).mockResolvedValue(done);
    const { result } = renderHook(() => useJob("j", 100));
    await waitFor(() => expect(result.current.offline).toBe(true));
    await waitFor(() => expect(result.current.job?.status).toBe("done"), { timeout: 3000 });
    expect(result.current.offline).toBe(false);
    expect(result.current.error).toBeNull();
    expect(getJob).toHaveBeenCalledTimes(3);
  });

  it("gives a clear message when the job no longer exists", async () => {
    const getJob = vi.spyOn(api, "getJob").mockRejectedValue(new ApiError(404, "yok"));
    const { result } = renderHook(() => useJob("j", 1));
    await waitFor(() => expect(result.current.error).toBe(text.errors.jobLost));
    expect(getJob).toHaveBeenCalledTimes(1);
  });

  it("tries again at once when the device comes back online", async () => {
    const getJob = vi.spyOn(api, "getJob").mockRejectedValueOnce(new ApiError(0, "ağ")).mockResolvedValue(done);
    const { result } = renderHook(() => useJob("j", 60000));
    await waitFor(() => expect(result.current.offline).toBe(true));
    act(() => {
      window.dispatchEvent(new Event("online"));
    });
    await waitFor(() => expect(result.current.job?.status).toBe("done"));
    expect(getJob).toHaveBeenCalledTimes(2);
  });
});
