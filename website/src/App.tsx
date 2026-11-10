// The Observatory page: one narrative scroll through MASTER_SPEC §8.2 A–H.
import { lazy, Suspense } from "react";
import { Status } from "./components/ClipParts";
import { loadRegistry } from "./lib/data";
import { useAsync } from "./lib/hooks";
import { Hero } from "./sections/Hero";
import { Lab } from "./sections/Lab";
import { RegistryContext } from "./sections/registry";

const WhyDiffusion = lazy(() => import("./sections/WhyDiffusion"));
const HeadToHead = lazy(() => import("./sections/HeadToHead"));
const InsidePolicy = lazy(() => import("./sections/InsidePolicy"));
const Findings = lazy(() => import("./sections/Findings"));
const BeyondPixels = lazy(() => import("./sections/BeyondPixels"));
const Methods = lazy(() => import("./sections/Methods"));

export const NAV = [
  ["why", "Why diffusion"],
  ["lab", "Lab"],
  ["compare", "Head to head"],
  ["inside", "Inside the policy"],
  ["findings", "Findings"],
  ["pixels", "Beyond pixels"],
  ["methods", "Methods"],
] as const;

function Topbar() {
  return (
    <header className="topbar">
      <a className="wordmark" href="#top">RoboForge<span>Observatory</span></a>
      <nav aria-label="Sections">
        <ul>{NAV.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}</ul>
      </nav>
    </header>
  );
}

const Loading = ({ what }: { what: string }) => (
  <section><div className="wrap"><Status state={{ status: "loading" }} what={what} /></div></section>
);

export function App() {
  const reg = useAsync(loadRegistry, []);
  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <Topbar />
      <main id="main">
        {reg.status !== "ready" ? (
          <section><div className="wrap"><Status state={reg} what="the Observatory data registry" /></div></section>
        ) : (
          <RegistryContext.Provider value={reg.value}>
            <Hero />
            <Suspense fallback={<Loading what="this section" />}>
              <WhyDiffusion />
              <Lab />
              <HeadToHead />
              <InsidePolicy />
              <Findings />
              <BeyondPixels />
              <Methods />
            </Suspense>
          </RegistryContext.Provider>
        )}
      </main>
      <footer>
        <div className="wrap">
          <p>RoboForge is a research project by Zackary Holt. Simulated robot control only; no physical robot was used.</p>
          <p>Built on robomimic, robosuite and MuJoCo; see Methods and credits for the full attribution and licences.</p>
        </div>
      </footer>
    </>
  );
}
