import { afterEach, describe, expect, it, vi } from "vitest";

import { text } from "../strings";
import { api, ApiError } from "./client";

const respond = (status: number, body: unknown) =>
  vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }));

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("sends json to the /api prefix", async () => {
    const fetchMock = respond(200, { job_id: "j1", position: 1 });
    vi.stubGlobal("fetch", fetchMock);
    await api.createDocument("p1", ["2101.03961"]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/documents");
    expect(JSON.parse(init.body)).toEqual({ plan_id: "p1", removed_ids: ["2101.03961"] });
  });

  it("uses the server's detail message", async () => {
    vi.stubGlobal("fetch", respond(429, { detail: "Bugünkü üretim hakkınızı kullandınız." }));
    await expect(api.getJob("x")).rejects.toEqual(new ApiError(429, "Bugünkü üretim hakkınızı kullandınız."));
  });

  it("turns validation errors and network failures into readable messages", async () => {
    vi.stubGlobal("fetch", respond(422, { detail: [{ loc: ["body", "topic"] }] }));
    await expect(api.search("x")).rejects.toThrow(text.errors.invalid);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(api.search("x")).rejects.toThrow(text.errors.network);
  });
});
