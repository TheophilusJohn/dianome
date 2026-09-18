// /writeup: docs/writeup.md rendered at build time (plugins/writeup.ts); every citation link points into the repo.
import { exists, html, modified, words } from "virtual:writeup";
import { $, nav } from "./shared";

nav("/writeup/");
$("writeup").innerHTML = html;
$("meta").textContent = exists ? `docs/writeup.md · ${words.toLocaleString()} words including tables · last modified ${modified ?? "?"} · every number links to the results file and heading it came from` : "docs/writeup.md is not written yet";
