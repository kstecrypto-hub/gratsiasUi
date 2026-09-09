import { expect, Page, test } from "@playwright/test";
import type { EvaluationDetail, HumanReference, ReferenceDraft } from "../../lib/types";

function recording() {
  const seconds = 12, rate = 8000, channels = 2;
  const data = Buffer.alloc(44 + seconds * rate * channels * 2);
  data.write("RIFF"); data.writeUInt32LE(data.length - 8, 4); data.write("WAVEfmt ", 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(channels, 22);
  data.writeUInt32LE(rate, 24); data.writeUInt32LE(rate * channels * 2, 28);
  data.writeUInt16LE(channels * 2, 32); data.writeUInt16LE(16, 34);
  data.write("data", 36); data.writeUInt32LE(data.length - 44, 40);
  for (let frame = 0; frame < seconds * rate; frame++) {
    data.writeInt16LE(Math.round(Math.sin(frame * 2 * Math.PI * 220 / rate) * 2000), 44 + frame * 4);
    data.writeInt16LE(Math.round(Math.sin(frame * 2 * Math.PI * 440 / rate) * 2000), 46 + frame * 4);
  }
  return data;
}

const emptyReference = (): HumanReference => ({
  quality: null, operator_channel: null, operator_channel_answered: false,
  expected_keywords: [], segments: [], verification_status: "not_started",
  verified_at: null, revision: "0".repeat(64),
});

const detail = (reference: HumanReference): EvaluationDetail => ({
  evaluation_id: "eval-001", split: "dev", mode: "stereo", duration_seconds: 12,
  metadata: { direction: "inbound", queue: "Sales", is_queue: true, transfer_state: "transferred",
    audio_topology: "Stereo", occurred_at: "2026-07-01T06:00:00Z",
    caller: "•••••••4567", callee: "•••••••6543", operators: [{ name: "PBX Operator", extension: "201" }] },
  reference,
});

async function setup(page: Page, options: { enabled?: boolean; reference?: HumanReference; conflict?: boolean } = {}) {
  let saved = options.reference || emptyReference();
  let saves = 0;
  let verifications = 0;
  const requests: string[] = [];
  await page.route("**/api/**", async (route) => {
    const request = route.request(), url = new URL(request.url());
    const path = url.pathname.replace(/^\/api/, ""), method = request.method();
    requests.push(`${method} ${path}`);
    let body: unknown;
    if (path === "/auth/me") body = { email: "admin@example.test" };
    else if (path === "/auth/csrf") body = { csrf_token: "test-csrf" };
    else if (path === "/features") body = { evaluation_ui_enabled: options.enabled !== false };
    else if (path === "/settings") body = { default_timezone: "Europe/Athens" };
    else if (path === "/evaluation") {
      let items = [
        { evaluation_id: "eval-001", split: "dev", mode: "stereo", duration_seconds: 12,
          direction: "inbound", quality: saved.quality, verification_status: saved.verification_status },
        { evaluation_id: "eval-002", split: "test", mode: "mono", duration_seconds: 20,
          direction: "outbound", quality: "clean", verification_status: "verified" },
      ];
      const filter = url.searchParams.get("filter");
      if (filter === "dev" || filter === "test") items = items.filter((item) => item.split === filter);
      if (filter === "verified") items = items.filter((item) => item.verification_status === "verified");
      if (filter === "unverified") items = items.filter((item) => item.verification_status !== "verified");
      body = { items, progress: { all: { verified: 1, total: 2 }, dev: { verified: 0, total: 1 }, test: { verified: 1, total: 1 } } };
    } else if (path === "/evaluation/keywords") {
      body = [{ id: "keyword-1", canonical_phrase: "προσφορά", category_name: "Sales" }];
    } else if (/^\/evaluation\/eval-001\/audio/.test(path)) {
      await route.fulfill({ contentType: "audio/wav", body: recording(), headers: { "Accept-Ranges": "bytes" } });
      return;
    } else if (path === "/evaluation/eval-001") body = detail(saved);
    else if (path === "/evaluation/eval-001/reference" && method === "PUT") {
      expect(request.headers()["x-csrf-token"]).toBe("test-csrf");
      saves += 1;
      if (options.conflict) {
        await route.fulfill({ status: 409, contentType: "application/json",
          body: JSON.stringify({ detail: "This reference changed in another tab. Reload before saving or verifying." }) });
        return;
      }
      const input = request.postDataJSON() as ReferenceDraft & { revision: string };
      expect(input.revision).toBe(saved.revision);
      saved = { ...input, verification_status: "in_progress", verified_at: null, revision: String(saves).repeat(64) };
      body = saved;
    } else if (path === "/evaluation/eval-001/verify" && method === "POST") {
      expect(request.headers()["x-csrf-token"]).toBe("test-csrf");
      expect(request.postDataJSON().revision).toBe(saved.revision);
      verifications += 1;
      saved = { ...saved, verification_status: "verified", verified_at: "2026-09-09T12:00:00Z", revision: "f".repeat(64) };
      body = saved;
    } else {
      await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Not found" }) });
      return;
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(body) });
  });
  return { requests, saved: () => saved, saves: () => saves, verifications: () => verifications };
}

async function openEditor(page: Page) {
  await page.goto("/evaluation/eval-001");
  await expect(page.getByRole("heading", { name: "Human reference segments" })).toBeVisible();
  await expect.poll(() => page.locator("audio").evaluate((element: HTMLAudioElement) => element.readyState)).toBeGreaterThanOrEqual(1);
}
const segment = (page: Page, index = 1) => page.getByRole("region", { name: `Reference segment ${index}`, exact: true });

test("evaluation reuses the authenticated shell, masks metadata, and never requests ASR", async ({ page }) => {
  const state = await setup(page);
  await openEditor(page);
  await expect(page.getByRole("navigation")).toContainText("Results");
  await expect(page.getByRole("navigation")).toContainText("Evaluation");
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();
  await expect(page.locator(".detail-grid")).toContainText("PBX Operator");
  await expect(page.locator(".detail-grid")).toContainText("•••••••4567");
  await expect(page.getByText("No reference text yet.", { exact: false })).toBeVisible();
  await expect(page.getByRole("textbox")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Transcript", exact: true })).toHaveCount(0);
  expect(state.requests.some((request) => request.includes("/calls/"))).toBe(false);
});

test("disabled evaluation hides navigation and editor without fetching data", async ({ page }) => {
  const state = await setup(page, { enabled: false });
  await page.goto("/evaluation/eval-001");
  await expect(page.getByText("Evaluation is unavailable in this environment.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Evaluation", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Add segment" })).toHaveCount(0);
  expect(state.requests.some((request) => request.includes("/evaluation"))).toBe(false);
});

test("evaluation dataset filters and verification counts", async ({ page }) => {
  await setup(page);
  await page.goto("/evaluation");
  await expect(page.getByRole("heading", { name: "1 / 2 verified" })).toBeVisible();
  await expect(page.getByText("DEV: 0 / 1")).toBeVisible();
  await expect(page.getByText("TEST: 1 / 1")).toBeVisible();
  const filters = page.getByRole("group", { name: "Dataset filters" });
  for (const [filter, present, absent] of [["DEV", "eval-001", "eval-002"], ["TEST", "eval-002", "eval-001"],
    ["Unverified", "eval-001", "eval-002"], ["Verified", "eval-002", "eval-001"]]) {
    await filters.getByRole("button", { name: filter, exact: true }).click();
    await expect(page.getByRole("link", { name: present })).toBeVisible();
    await expect(page.getByRole("link", { name: absent })).toHaveCount(0);
  }
  await filters.getByRole("button", { name: "All", exact: true }).click();
  await expect(page.getByRole("link", { name: "eval-001" })).toBeVisible();
  await expect(page.getByRole("link", { name: "eval-002" })).toBeVisible();
});

test("real audio plays, pauses, seeks and keeps position and speed across channels", async ({ page }) => {
  await setup(page);
  await openEditor(page);
  const audio = page.locator("audio");
  await expect(audio).toHaveJSProperty("controls", true);
  await audio.evaluate((element: HTMLAudioElement) => element.play());
  await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeGreaterThan(0);
  await audio.evaluate((element: HTMLAudioElement) => element.pause());
  await audio.evaluate((element: HTMLAudioElement) => { element.currentTime = 6; });
  for (const [name, position] of [["Seek back 2 seconds", 4], ["Seek forward 5 seconds", 9],
    ["Seek back 5 seconds", 4], ["Seek forward 2 seconds", 6]] as const) {
    await page.getByRole("button", { name }).click();
    await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeCloseTo(position, 1);
  }
  await page.getByLabel("Playback speed").selectOption("1.25");
  await expect(audio).toHaveJSProperty("playbackRate", 1.25);
  for (const [name, path] of [["Channel A", "/channel/0"], ["Channel B", "/channel/1"], ["Stereo", ""]] as const) {
    await page.getByRole("button", { name, exact: true }).click();
    await expect(audio).toHaveAttribute("src", "/api/evaluation/eval-001/audio" + path);
    await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.readyState)).toBeGreaterThanOrEqual(1);
    await expect.poll(() => audio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeCloseTo(6, 1);
    await expect(audio).toHaveJSProperty("playbackRate", 1.25);
  }
});

test("add edit delete, player timestamps, overlap, entities and verification round-trip", async ({ page }) => {
  const state = await setup(page);
  await openEditor(page);
  const mark = page.getByRole("button", { name: "Mark fully reviewed" });
  await expect(mark).toBeDisabled();
  await page.getByLabel("Quality label").selectOption("noisy");
  await page.getByLabel("Human operator channel").selectOption("1");
  await page.getByRole("button", { name: "Add segment", exact: true }).click();
  await segment(page).getByLabel("Speaker", { exact: true }).selectOption("Operator");
  await segment(page).getByLabel("Channel", { exact: true }).selectOption("1");
  await segment(page).getByLabel("Exact reference transcript").fill("Καλημέρα, προσφορά, εε προσφορά.");
  const audio = page.locator("audio");
  await audio.evaluate((element: HTMLAudioElement) => { element.currentTime = 1.25; });
  await segment(page).getByRole("button", { name: "Set start to player position" }).click();
  await expect(segment(page).getByLabel("Start time (seconds)")).toHaveValue("1.25");
  await audio.evaluate((element: HTMLAudioElement) => { element.currentTime = 4.5; });
  await segment(page).getByRole("button", { name: "Set end to player position" }).click();
  await expect(segment(page).getByLabel("End time (seconds)")).toHaveValue("4.5");
  for (const [name, position] of [["Seek to start", 1.25], ["Seek to end", 4.5]] as const) {
    await segment(page).getByRole("button", { name, exact: true }).click();
    await audio.evaluate((element: HTMLAudioElement) => element.pause());
    expect(await audio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeCloseTo(position, 0);
  }
  await segment(page).locator(".speaker button").click();
  await audio.evaluate((element: HTMLAudioElement) => element.pause());
  expect(await audio.evaluate((element: HTMLAudioElement) => element.currentTime)).toBeCloseTo(1.25, 0);
  await segment(page).getByText("Entity annotations", { exact: true }).click();
  for (const [label, value] of [["Names", "Test Name"], ["Telephone numbers", "2100000000"],
    ["Licence plates", "ABC-1234"], ["Vehicle models", "Example Model"]]) {
    await segment(page).getByLabel(label, { exact: true }).fill(value);
    await segment(page).getByRole("button", { name: `Add ${label.toLowerCase()}`, exact: true }).click();
  }
  await page.getByRole("checkbox", { name: "προσφορά (Sales)" }).check();
  await page.getByRole("button", { name: "Add segment", exact: true }).click();
  await segment(page, 2).getByLabel("Start time (seconds)").fill("2");
  await segment(page, 2).getByLabel("End time (seconds)").fill("5");
  await segment(page, 2).getByLabel("Exclude unintelligible region from WER").check();
  await expect(segment(page, 2)).toHaveClass(/excluded/);
  await page.getByRole("button", { name: "Add segment", exact: true }).click();
  await segment(page, 3).getByRole("button", { name: "Delete segment" }).click();
  await expect(page.locator(".reference-segment")).toHaveCount(2);
  await expect(mark).toBeDisabled(); // No unsaved edits may be verified.
  await page.getByRole("button", { name: "Save reference", exact: true }).click();
  await expect(page.getByText("Reference saved.", { exact: true })).toBeVisible();
  expect(state.saved().segments[0].entities).toEqual({
    names: ["Test Name"], telephone_numbers: ["2100000000"], licence_plates: ["ABC-1234"], vehicle_models: ["Example Model"],
  });
  expect(state.saved().operator_channel).toBe(1);
  expect(state.saved().quality).toBe("noisy");
  expect(state.saved().expected_keywords).toEqual(["προσφορά"]);
  expect(state.saved().segments[1].exclude_from_wer).toBe(true);
  expect(state.saved().segments[1].text).toBe("");
  await expect(mark).toBeEnabled();
  await mark.click();
  await expect(page.getByText("Reference marked fully reviewed.")).toBeVisible();
  expect(state.verifications()).toBe(1);
  await page.reload();
  await expect(segment(page).getByLabel("Exact reference transcript")).toHaveValue("Καλημέρα, προσφορά, εε προσφορά.");
  await expect(mark).toBeDisabled();
  await segment(page).getByLabel("Exact reference transcript").fill("Exact audible correction");
  await expect(page.getByText("Editing requires a new explicit review.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Save reference", exact: true }).click();
  await expect.poll(() => state.saved().verification_status).toBe("in_progress");
  expect(state.saved().verified_at).toBeNull();
  await expect(mark).toBeEnabled();
  expect(state.requests.some((request) => /\/calls\//.test(request))).toBe(false);
});

test("invalid times and missing stereo answer prevent verification; unknown is explicit", async ({ page }) => {
  await setup(page);
  await openEditor(page);
  await page.getByRole("button", { name: "Add segment", exact: true }).click();
  await page.getByLabel("Quality label").selectOption("normal");
  await segment(page).getByLabel("Exact reference transcript").fill("Audible words");
  await segment(page).getByLabel("End time (seconds)").fill("13");
  await page.getByRole("button", { name: "Save reference", exact: true }).click();
  await expect(page.getByRole("button", { name: "Mark fully reviewed" })).toBeDisabled();
  await expect(page.getByText("Segment 1: enter valid start/end times within the recording.")).toBeVisible();
  await segment(page).getByLabel("End time (seconds)").fill("3");
  await page.getByRole("button", { name: "Save reference", exact: true }).click();
  await expect(page.getByRole("button", { name: "Mark fully reviewed" })).toBeDisabled();
  await page.getByLabel("Human operator channel").selectOption("unknown");
  await page.getByRole("button", { name: "Save reference", exact: true }).click();
  await expect(page.getByRole("button", { name: "Mark fully reviewed" })).toBeEnabled();
});

test("save conflicts preserve the current draft", async ({ page }) => {
  const state = await setup(page, { conflict: true });
  await openEditor(page);
  await page.getByRole("button", { name: "Add segment", exact: true }).click();
  await segment(page).getByLabel("Exact reference transcript").fill("Keep this unsaved work");
  await page.getByRole("button", { name: "Save reference", exact: true }).click();
  await expect(page.locator(".form-error[role='alert']")).toContainText("changed in another tab");
  await expect(segment(page).getByLabel("Exact reference transcript")).toHaveValue("Keep this unsaved work");
  expect(state.saved().segments).toEqual([]);
});
