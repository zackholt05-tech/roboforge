// Entry: the Observatory page, or the bare verification renderer (?verify=...) that
// `make web-verify` compares with MuJoCo's segmentation renders.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/site.css";

const params = new URLSearchParams(window.location.search);
const root = createRoot(document.getElementById("root")!);
if (params.has("verify")) {
  import("./verify/VerifyPage").then(({ VerifyPage }) => root.render(<VerifyPage params={params} />));
} else {
  root.render(<StrictMode><App /></StrictMode>);
}
