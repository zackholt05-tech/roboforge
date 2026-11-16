// Display helpers added in Session 20 (D-136): they only word or round values that the data holds.
import { describe, expect, it } from "vitest";
import { cm, PARTITION, startLabel } from "./format";

describe("startLabel", () => {
  it("keeps the recorded partition's wording for every frozen suite", () => {
    for (const [partition, words] of Object.entries(PARTITION)) {
      expect(startLabel(partition, "can_final_test_0000")).toBe(words);
    }
  });
  it("names the X2 held-out suite, which was recorded under eval_adhoc (D-123)", () => {
    expect(startLabel("eval_adhoc", "can_x2_heldout_0010")).toBe("X2 held-out start");
    expect(startLabel("eval_adhoc", "can_dev_0001")).toBe(PARTITION.eval_adhoc);
  });
  it("falls back to the raw partition name when it is unknown", () => {
    expect(startLabel("eval_other", "lift_x_0000")).toBe("eval_other");
  });
});

describe("cm", () => {
  it("shows a length in metres as centimetres without float noise", () => {
    expect(cm(0.07)).toBe("7"); // 0.07 * 100 is 7.000000000000001 in floating point
    expect(cm(0.055)).toBe("5.5");
  });
});
