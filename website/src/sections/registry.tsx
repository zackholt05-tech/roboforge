// The export registry (data/registry.json) as React context, plus role lookups. Clip roles come
// from the D-089 selection rules (e.g. "hero", "gallery:first_failure", "h2h:can:b_only_diffusion").
import { createContext, useContext } from "react";
import type { Registry, RegistryClip } from "../lib/schema";

export const RegistryContext = createContext<Registry | null>(null);

export function useRegistry(): Registry {
  const r = useContext(RegistryContext);
  if (!r) throw new Error("registry not loaded");
  return r;
}

export function byRole(reg: Registry, role: string, family?: string): RegistryClip | undefined {
  return reg.clips.find((c) => c.roles.includes(role) && (!family || c.family === family));
}

export const ROLE_LABEL: Record<string, string> = {
  "gallery:first_success": "first success",
  "gallery:first_failure": "first failure",
};

export function roleText(c: RegistryClip): string {
  const g = c.roles.filter((r) => r.startsWith("gallery:"));
  const parts = g.map((r) => ROLE_LABEL[r] ?? `first ${r.replace("gallery:category:", "").replaceAll("_", " ")}`);
  return [...new Set(parts)].join(", ");
}
